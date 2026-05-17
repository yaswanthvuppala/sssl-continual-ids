import tensorflow as tf
from tqdm import tqdm
import os
import json
import numpy as np

class FixMatchTrainer:
    """
    FixMatch Trainer for Semi-Supervised Learning.
    Applies consistency regularization between weakly and strongly augmented unlabeled views,
    combined with a standard supervised loss on a small set of labeled examples.

    Improvements for recall:
      - Focal Loss to downweight easy/majority-class samples
      - Per-class weighting to handle class imbalance
      - Lower confidence threshold to include more hard pseudo-labels
    """
    def __init__(self, encoder: tf.keras.Model, head: tf.keras.Model, gpm=None, 
                 lr: float = 0.03, confidence_threshold: float = 0.90,
                 class_weights: dict = None, focal_gamma: float = 2.0):
        self.encoder = encoder
        self.head = head
        self.gpm = gpm
        self.confidence_threshold = confidence_threshold
        self.class_weights = class_weights  # {0: w0, 1: w1, ...}
        self.focal_gamma = focal_gamma
        
        # Ensure encoder is frozen
        self.encoder.trainable = False
        
        # SGD with momentum is standard for FixMatch
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, head=self.head)

    @tf.function
    def _encode(self, x):
        return self.encoder(x, training=False)

    def _focal_weighted_loss(self, y_true, logits, sample_weight=None):
        """
        Focal Loss with optional per-sample class weighting.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        ce_loss = self.loss_fn(y_true, logits)  # per-sample cross-entropy

        # Focal modulation
        probs = tf.nn.softmax(logits, axis=-1)
        y_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=logits.shape[-1])
        p_t = tf.reduce_sum(probs * y_one_hot, axis=-1)  # probability of true class
        focal_weight = tf.pow(1.0 - p_t, self.focal_gamma)

        loss = focal_weight * ce_loss

        # Class weighting
        if sample_weight is not None:
            loss = loss * sample_weight

        return loss

    @tf.function
    def _compute_gradients(self, x_l, y_l, x_u_weak, x_u_strong, class_weight_tensor, lambda_u: float = 1.0):
        """
        Forward pass + loss + gradients (runs inside tf.function graph for speed).
        GPM projection is intentionally excluded — it needs eager .numpy() calls.
        """
        # 1. Pseudo-label generation (no gradient)
        emb_u_weak = self._encode(x_u_weak)
        logits_weak = self.head(emb_u_weak, training=False)
        probs_weak = tf.nn.softmax(logits_weak)
        
        max_probs = tf.reduce_max(probs_weak, axis=-1)
        pseudo_labels = tf.argmax(probs_weak, axis=-1)
        mask = tf.cast(max_probs >= self.confidence_threshold, tf.float32)
        
        with tf.GradientTape() as tape:
            # Supervised path — focal loss with class weighting
            emb_l = self._encode(x_l)
            logits_l = self.head(emb_l, training=True)

            # Build per-sample weights from class weights
            sw_l = tf.gather(class_weight_tensor, tf.cast(y_l, tf.int32))
            focal_loss_l = self._focal_weighted_loss(y_l, logits_l, sample_weight=sw_l)
            loss_s = tf.reduce_mean(focal_loss_l)
            
            # Unsupervised consistency path (standard CE, no class weighting)
            emb_u_strong = self._encode(x_u_strong)
            logits_strong = self.head(emb_u_strong, training=True)
            loss_u_per_sample = self.loss_fn(pseudo_labels, logits_strong)
            loss_u = tf.reduce_mean(loss_u_per_sample * mask)
            
            total_loss = loss_s + lambda_u * loss_u
            
        grads = tape.gradient(total_loss, self.head.trainable_variables)
        return grads, loss_s, loss_u, total_loss, tf.reduce_mean(mask)

    def train_step(self, x_l, y_l, x_u_weak, x_u_strong, class_weight_tensor, lambda_u: float = 1.0):
        """
        Single FixMatch training step (eager mode — allows GPM numpy projection).
        """
        grads, loss_s, loss_u, total_loss, mask_rate = self._compute_gradients(
            x_l, y_l, x_u_weak, x_u_strong, class_weight_tensor, lambda_u
        )
        
        # GPM gradient projection runs in eager mode (needs .numpy())
        if self.gpm is not None:
            grads = self.gpm.project_gradients(grads, self.head.trainable_variables)
            
        self.optimizer.apply_gradients(zip(grads, self.head.trainable_variables))
        
        return {
            "loss_s": loss_s,
            "loss_u": loss_u,
            "total_loss": total_loss,
            "mask_rate": mask_rate
        }

    def train(self, labeled_ds: tf.data.Dataset, unlabeled_ds: tf.data.Dataset, 
              task_name: str, epochs: int = 10, lambda_u: float = 1.0):
        """
        Training loop for a specific task.
        Returns training history dict for visualization.
        """
        print(f"Starting FixMatch training for task: {task_name}")
        writer = tf.summary.create_file_writer(f'./logs/task_{task_name}')
        ckpt_manager = tf.train.CheckpointManager(self.checkpoint, f'./checkpoints/{task_name}', max_to_keep=1)
        
        # Prepare class weight tensor
        if self.class_weights is not None:
            num_classes = self.head.output_shape[-1]
            cw_array = np.ones(num_classes, dtype=np.float32)
            for cls_id, w in self.class_weights.items():
                if cls_id < num_classes:
                    cw_array[cls_id] = w
            class_weight_tensor = tf.constant(cw_array, dtype=tf.float32)
        else:
            num_classes = self.head.output_shape[-1]
            class_weight_tensor = tf.ones(num_classes, dtype=tf.float32)

        # Create an iterator for the unlabeled dataset (usually much larger)
        unlabeled_iter = iter(unlabeled_ds.repeat())
        
        # History tracking for visualization
        history = {"loss_s": [], "loss_u": [], "total_loss": [], "mask_rate": []}
        
        for epoch in range(epochs):
            metrics = {"loss_s": 0.0, "loss_u": 0.0, "total_loss": 0.0, "mask_rate": 0.0}
            steps = 0
            
            pbar = tqdm(labeled_ds, desc=f"Epoch {epoch+1}/{epochs}")
            for x_l, y_l in pbar:
                # Get a batch from unlabeled stream
                x_u_weak, x_u_strong = next(unlabeled_iter)
                
                step_metrics = self.train_step(x_l, y_l, x_u_weak, x_u_strong, class_weight_tensor, lambda_u)
                
                for k, v in step_metrics.items():
                    metrics[k] += float(v)
                steps += 1
                
                pbar.set_postfix({
                    "Ls": f"{float(step_metrics['loss_s']):.3f}", 
                    "Lu": f"{float(step_metrics['loss_u']):.3f}",
                    "Mask": f"{float(step_metrics['mask_rate']):.2f}"
                })
                
            avg_metrics = {k: v / max(1, steps) for k, v in metrics.items()}
            print(f"Epoch {epoch+1} summary: {avg_metrics}")
            
            # Record history
            for k in history:
                history[k].append(avg_metrics[k])
            
            with writer.as_default():
                for k, v in avg_metrics.items():
                    tf.summary.scalar(k, v, step=epoch)
                    
            ckpt_manager.save()
            
        print(f"Task {task_name} training complete.")

        # Save training history for visualization
        history_dir = f"./logs/task_{task_name}"
        os.makedirs(history_dir, exist_ok=True)
        history_path = os.path.join(history_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {history_path}")

        return avg_metrics
