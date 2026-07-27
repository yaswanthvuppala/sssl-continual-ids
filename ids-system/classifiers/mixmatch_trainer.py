import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os


class MixMatchTrainer:
    """
    MixMatch Trainer for Semi-Supervised Learning.
    Interpolates labeled and unlabeled representations using MixUp after
    generating sharpened pseudo-labels from multiple augmented views.
    """
    def __init__(self, encoder: tf.keras.Model, head: tf.keras.Model, gpm=None,
                 lr: float = 0.03, temperature: float = 0.5, alpha: float = 0.75,
                 num_augments: int = 2, lambda_u: float = 1.0,
                 log_dir: str = None):
        self.encoder = encoder
        self.head = head
        self.gpm = gpm
        self.temperature = temperature  # for label sharpening
        self.alpha = alpha              # Beta distribution parameter for MixUp
        self.num_augments = num_augments
        self.lambda_u = lambda_u
        self.log_dir = log_dir

        self.encoder.trainable = False
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)

    @tf.function
    def _encode(self, x):
        return self.encoder(x, training=False)

    def _sharpen(self, probs: tf.Tensor) -> tf.Tensor:
        """Temperature sharpening of probability distribution."""
        temp = max(self.temperature, 1e-4)
        p = tf.pow(tf.maximum(probs, 1e-7), 1.0 / temp)
        return p / tf.maximum(tf.reduce_sum(p, axis=-1, keepdims=True), 1e-7)

    def _mixup(self, x1: tf.Tensor, x2: tf.Tensor, y1: tf.Tensor, y2: tf.Tensor):
        """MixUp two batches of (features, labels)."""
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1.0 - lam)  # ensure lam >= 0.5
        lam_tf = tf.constant(lam, dtype=tf.float32)
        x_mix = lam_tf * x1 + (1.0 - lam_tf) * x2
        y_mix = lam_tf * y1 + (1.0 - lam_tf) * y2
        return x_mix, y_mix

    @tf.function
    def train_step(self, emb_l_mix, y_l_mix, emb_u_mix, pseudo_u_mix):
        """Single MixMatch training step on pre-computed mixed embeddings."""
        with tf.GradientTape() as tape:
            # Labeled MixUp loss — cross entropy on mixed target
            logits_l = self.head(emb_l_mix, training=True)
            loss_s = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y_l_mix, logits_l, from_logits=True)
            )

            # Unlabeled MixUp loss — MSE on mixed sharpened pseudo-labels
            logits_u = self.head(emb_u_mix, training=True)
            probs_u = tf.nn.softmax(logits_u)
            loss_u = tf.reduce_mean(tf.square(probs_u - pseudo_u_mix))

            total_loss = loss_s + self.lambda_u * loss_u

        grads = tape.gradient(total_loss, self.head.trainable_variables)
        if self.gpm is not None:
            grads = self.gpm.project_gradients(grads, self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.head.trainable_variables))

        return {"loss_s": loss_s, "loss_u": loss_u, "total_loss": total_loss}

    def train(self, labeled_ds: tf.data.Dataset, unlabeled_ds: tf.data.Dataset,
              task_name: str, num_classes: int = 2, epochs: int = 10):
        """Training loop for MixMatch."""
        print(f"Starting MixMatch training for task: {task_name}")
        log_path = self.log_dir or f'./logs/mixmatch_{task_name}'
        writer = tf.summary.create_file_writer(log_path)
        unlabeled_iter = iter(unlabeled_ds.repeat())

        for epoch in range(epochs):
            metrics = {"loss_s": 0.0, "loss_u": 0.0, "total_loss": 0.0}
            steps = 0

            pbar = tqdm(labeled_ds, desc=f"Epoch {epoch+1}/{epochs}")
            for x_l, y_l in pbar:
                # Encode labeled
                emb_l = self._encode(x_l)
                y_l_onehot = tf.one_hot(y_l, depth=num_classes)

                # Encode unlabeled — average predictions over augmented views
                x_u_weak, x_u_strong = next(unlabeled_iter)
                emb_u1 = self._encode(x_u_weak)
                emb_u2 = self._encode(x_u_strong)
                emb_u = (emb_u1 + emb_u2) / 2.0  # average embeddings

                # Generate sharpened pseudo-labels
                probs_u = tf.nn.softmax(self.head(emb_u, training=False))
                pseudo = self._sharpen(probs_u)

                # MixUp: Combine labeled and unlabeled pools, shuffle, and apply MixUp
                all_emb = tf.concat([emb_l, emb_u], axis=0)
                all_targets = tf.concat([y_l_onehot, pseudo], axis=0)
                indices = tf.random.shuffle(tf.range(tf.shape(all_emb)[0]))
                all_emb_shuffled = tf.gather(all_emb, indices)
                all_targets_shuffled = tf.gather(all_targets, indices)

                n_labeled = tf.shape(emb_l)[0]
                emb_l_mix, y_l_mix = self._mixup(
                    emb_l, all_emb_shuffled[:n_labeled],
                    y_l_onehot, all_targets_shuffled[:n_labeled]
                )
                emb_u_mix, pseudo_u_mix = self._mixup(
                    emb_u, all_emb_shuffled[n_labeled:],
                    pseudo, all_targets_shuffled[n_labeled:]
                )

                step_metrics = self.train_step(emb_l_mix, y_l_mix, emb_u_mix, pseudo_u_mix)
                for k, v in step_metrics.items():
                    metrics[k] += float(v)
                steps += 1

                pbar.set_postfix({
                    "Ls": f"{float(step_metrics['loss_s']):.3f}",
                    "Lu": f"{float(step_metrics['loss_u']):.3f}"
                })

            avg_metrics = {k: v / max(1, steps) for k, v in metrics.items()}
            print(f"Epoch {epoch+1} summary: {avg_metrics}")

            with writer.as_default():
                for k, v in avg_metrics.items():
                    tf.summary.scalar(k, v, step=epoch)

        print(f"MixMatch task {task_name} complete.")
        return avg_metrics

