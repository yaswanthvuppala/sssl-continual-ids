import tensorflow as tf
from tqdm import tqdm
import os

class FixMatchTrainer:
    """
    FixMatch Trainer for Semi-Supervised Learning.
    Applies consistency regularization between weakly and strongly augmented unlabeled views,
    combined with a standard supervised loss on a small set of labeled examples.
    """
    def __init__(self, encoder: tf.keras.Model, head: tf.keras.Model, gpm=None, 
                 lr: float = 0.03, confidence_threshold: float = 0.95):
        self.encoder = encoder
        self.head = head
        self.gpm = gpm
        self.confidence_threshold = confidence_threshold
        
        # Ensure encoder is frozen
        self.encoder.trainable = False
        
        # SGD with momentum is standard for FixMatch
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, head=self.head)

    @tf.function
    def _encode(self, x):
        return self.encoder(x, training=False)

    @tf.function
    def train_step(self, x_l, y_l, x_u_weak, x_u_strong, lambda_u: float = 1.0):
        """
        Single FixMatch training step.
        """
        # 1. Pseudo-label generation (no gradient)
        emb_u_weak = self._encode(x_u_weak)
        logits_weak = self.head(emb_u_weak, training=False)
        probs_weak = tf.nn.softmax(logits_weak)
        
        max_probs = tf.reduce_max(probs_weak, axis=-1)
        pseudo_labels = tf.argmax(probs_weak, axis=-1)
        mask = tf.cast(max_probs >= self.confidence_threshold, tf.float32)
        
        with tf.GradientTape() as tape:
            # Supervised path
            emb_l = self._encode(x_l)
            logits_l = self.head(emb_l, training=True)
            loss_s = tf.reduce_mean(self.loss_fn(y_l, logits_l))
            
            # Unsupervised consistency path
            emb_u_strong = self._encode(x_u_strong)
            logits_strong = self.head(emb_u_strong, training=True)
            loss_u_per_sample = self.loss_fn(pseudo_labels, logits_strong)
            loss_u = tf.reduce_mean(loss_u_per_sample * mask)
            
            total_loss = loss_s + lambda_u * loss_u
            
        grads = tape.gradient(total_loss, self.head.trainable_variables)
        
        # Project gradients through GPM if it exists and has stored bases
        if self.gpm is not None:
            grads = self.gpm.project_gradients(grads, self.head.trainable_variables)
            
        self.optimizer.apply_gradients(zip(grads, self.head.trainable_variables))
        
        return {
            "loss_s": loss_s,
            "loss_u": loss_u,
            "total_loss": total_loss,
            "mask_rate": tf.reduce_mean(mask)
        }

    def train(self, labeled_ds: tf.data.Dataset, unlabeled_ds: tf.data.Dataset, 
              task_name: str, epochs: int = 10, lambda_u: float = 1.0):
        """
        Training loop for a specific task.
        """
        print(f"Starting FixMatch training for task: {task_name}")
        writer = tf.summary.create_file_writer(f'./logs/task_{task_name}')
        ckpt_manager = tf.train.CheckpointManager(self.checkpoint, f'./checkpoints/{task_name}', max_to_keep=1)
        
        # Create an iterator for the unlabeled dataset (usually much larger)
        unlabeled_iter = iter(unlabeled_ds.repeat())
        
        for epoch in range(epochs):
            metrics = {"loss_s": 0.0, "loss_u": 0.0, "total_loss": 0.0, "mask_rate": 0.0}
            steps = 0
            
            pbar = tqdm(labeled_ds, desc=f"Epoch {epoch+1}/{epochs}")
            for x_l, y_l in pbar:
                # Get a batch from unlabeled stream
                x_u_weak, x_u_strong = next(unlabeled_iter)
                
                # Match batch sizes if needed (FixMatch typically uses a ratio like 1:7)
                # For simplicity here, we assume they are provided in compatible batches
                
                step_metrics = self.train_step(x_l, y_l, x_u_weak, x_u_strong, lambda_u)
                
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
            
            with writer.as_default():
                for k, v in avg_metrics.items():
                    tf.summary.scalar(k, v, step=epoch)
                    
            ckpt_manager.save()
            
        print(f"Task {task_name} training complete.")
        return avg_metrics
