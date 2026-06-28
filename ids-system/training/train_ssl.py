import os
import sys
import argparse
import tensorflow as tf
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from data.tf_dataset import make_unlabeled_dataset
from encoder.flow_encoder import build_flow_encoder
from encoder.projection_head import build_projection_head
from encoder.losses import nt_xent_loss

class SSLPretrainer:
    def __init__(self, input_dim: int, hidden_dim: int = 512, embed_dim: int = 256, proj_dim: int = 128,
                 ckpt_dir: str = None, log_dir: str = None):
        self.encoder = build_flow_encoder(input_dim, hidden_dim, embed_dim)
        self.projector = build_projection_head(embed_dim, proj_dim)
        self.temperature = 0.1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        
        self.trainable_vars = self.encoder.trainable_variables + self.projector.trainable_variables
        self.log_dir = log_dir or './logs/ssl'
        
        # Setup checkpointing
        ssl_ckpt_dir = ckpt_dir or './checkpoints/ssl'
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, projector=self.projector)
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, ssl_ckpt_dir, max_to_keep=3)

    @tf.function
    def train_step(self, x1, x2):
        with tf.GradientTape() as tape:
            z1 = self.projector(self.encoder(x1, training=True), training=True)
            z2 = self.projector(self.encoder(x2, training=True), training=True)
            loss = nt_xent_loss(z1, z2, temperature=self.temperature)
            
        grads = tape.gradient(loss, self.trainable_vars)
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        return loss

    def train(self, dataset: tf.data.Dataset, epochs: int = 10):
        print(f"Starting SSL Pretraining for {epochs} epochs...")
        
        # Create summary writer for TensorBoard
        writer = tf.summary.create_file_writer(self.log_dir)
        
        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            
            pbar = tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}")
            for x1, x2 in pbar:
                loss = self.train_step(x1, x2)
                total_loss += float(loss)
                steps += 1
                pbar.set_postfix({"loss": f"{loss:.4f}"})
                
            avg_loss = total_loss / max(1, steps)
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
            
            with writer.as_default():
                tf.summary.scalar('ssl_loss', avg_loss, step=epoch)
                
            # Save checkpoint
            self.ckpt_manager.save()
            
    def save_frozen_encoder(self, path: str = "./checkpoints/encoder_frozen.keras"):
        self.encoder.trainable = False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.encoder.save(path)
        print(f"Frozen encoder saved to {path}")

def main():
    parser = argparse.ArgumentParser(description="SSL Pretraining")
    parser.add_argument("--epochs", type=int, default=10, help="Number of pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--train_csv", type=str, default=None, help="Training CSV for real dataset pretraining")
    parser.add_argument("--dataset", type=str, choices=["cicids2017", "kddcup99", "unsw"],
                        default=None, help="Load a supported raw dataset")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Dataset directory or raw data file")
    parser.add_argument("--label_col", type=str, default="Label", help="Label column in the training CSV")
    parser.add_argument("--preprocessor_path", type=str, default=None,
                        help="Where to save the fitted preprocessor")
    parser.add_argument("--dataset_name", type=str, default="default",
                        help="Dataset identifier for scoping output paths")
    args = parser.parse_args()

    # Resolve dataset-scoped base paths
    ds = args.dataset_name
    ckpt_base = f"./checkpoints/{ds}"
    log_base = f"./logs/{ds}"
    os.makedirs(ckpt_base, exist_ok=True)
    os.makedirs(log_base, exist_ok=True)

    if args.preprocessor_path is None:
        args.preprocessor_path = f"{ckpt_base}/preprocessor.pkl"

    loader = FlowDatasetLoader(data_path=args.data_path or ".")
    if args.dataset:
        if not args.data_path:
            raise ValueError("--data_path is required when --dataset is used")
        df = loader.load_dataset(
            args.dataset, split="train", label_col=args.label_col
        )
    elif args.train_csv:
        df = loader.load_csv(args.train_csv, label_col=args.label_col)
    else:
        df = loader.create_synthetic_data(num_samples=50000, num_features=80)
    
    preprocessor = FlowPreprocessor()
    features, _ = preprocessor.fit_transform(df, label_col=args.label_col)
    preprocessor.save(args.preprocessor_path)
    print(f"Fitted preprocessor saved to {args.preprocessor_path}")
    
    dataset = make_unlabeled_dataset(features, batch_size=args.batch_size, for_ssl=True)
    
    input_dim = features.shape[1]
    
    pretrainer = SSLPretrainer(
        input_dim=input_dim,
        ckpt_dir=f"{ckpt_base}/ssl",
        log_dir=f"{log_base}/ssl",
    )
    pretrainer.train(dataset, epochs=args.epochs)
    pretrainer.save_frozen_encoder(f"{ckpt_base}/encoder_frozen.keras")

if __name__ == "__main__":
    main()
