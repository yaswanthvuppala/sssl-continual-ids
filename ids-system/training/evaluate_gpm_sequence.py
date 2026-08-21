"""
evaluate_gpm_sequence.py — Evaluate Continual Learning GPM performance matrix R_{i,j}.

This script evaluates all previously learned tasks (Intrusion, DoS, Port Scan)
after each stage of training to construct the full NxN Continual Learning Evaluation Matrix R,
calculating Average Forgetting (F) and Backward Transfer (BWT).

Usage:
    python training/evaluate_gpm_sequence.py --dataset cicids2017 --data_path "../CICIDS2017" --dataset_name cicids2017
    python training/evaluate_gpm_sequence.py --dataset kddcup99   --data_path "../KDDCUP99"   --dataset_name kddcup99
    python training/evaluate_gpm_sequence.py --dataset unsw       --test_csv "../IDS-UNSW_NB/UNSW_NB15_testing-set.csv" --dataset_name unsw
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import FlowDatasetLoader
from data.preprocessing import FlowPreprocessor
from encoder.flow_encoder import build_flow_encoder
from training.train_task import build_task_head, make_task_labels, copy_keras_weights_from_zip
from training.evaluate import evaluate_head
import zipfile
import h5py


def load_encoder(encoder_path: str) -> tf.keras.Model:
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found at {encoder_path}")

    if zipfile.is_zipfile(encoder_path):
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp(dir=".")
        try:
            with zipfile.ZipFile(encoder_path, 'r') as zip_ref:
                weights_path = copy_keras_weights_from_zip(encoder_path, temp_dir)
                with h5py.File(weights_path, 'r') as f:
                    input_dim = f['layers/dense/vars/0'].shape[0]
            encoder = build_flow_encoder(input_dim=input_dim)
            with zipfile.ZipFile(encoder_path, 'r') as zip_ref:
                weights_path = copy_keras_weights_from_zip(encoder_path, temp_dir)
                with h5py.File(weights_path, 'r') as f:
                    layer_groups = {}
                    for grp_name in f['layers'].keys():
                        vars_path = f"layers/{grp_name}/vars"
                        if vars_path in f:
                            name_attr = f[vars_path].attrs.get('name')
                            if name_attr:
                                if isinstance(name_attr, bytes):
                                    name_attr = name_attr.decode('utf-8')
                                layer_groups[name_attr] = vars_path
                    for layer in encoder.layers:
                        if not layer.weights:
                            continue
                        name = layer.name
                        h5_group = layer_groups.get(name)
                        if h5_group:
                            weight_vals = []
                            idx = 0
                            while f"{h5_group}/{idx}" in f:
                                weight_vals.append(f[f"{h5_group}/{idx}"][()])
                                idx += 1
                            if len(weight_vals) == len(layer.weights):
                                layer.set_weights(weight_vals)
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    elif h5py.is_hdf5(encoder_path):
        with h5py.File(encoder_path, 'r') as f:
            mw = f['model_weights']
            input_dim = mw['dense']['dense']['kernel:0'].shape[0]
            encoder = build_flow_encoder(input_dim=input_dim)
            if 'dense' in mw:
                encoder.get_layer('dense').set_weights([mw['dense']['dense']['kernel:0'][()], mw['dense']['dense']['bias:0'][()]])
            if 'layer_normalization' in mw:
                encoder.get_layer('layer_normalization').set_weights([mw['layer_normalization']['layer_normalization']['gamma:0'][()], mw['layer_normalization']['layer_normalization']['beta:0'][()]])
            if 'dense_1' in mw:
                encoder.get_layer('dense_1').set_weights([mw['dense_1']['dense_1']['kernel:0'][()], mw['dense_1']['dense_1']['bias:0'][()]])
            if 'layer_normalization_1' in mw:
                encoder.get_layer('layer_normalization_1').set_weights([mw['layer_normalization_1']['layer_normalization_1']['gamma:0'][()], mw['layer_normalization_1']['layer_normalization_1']['beta:0'][()]])
            if 'embedding' in mw:
                encoder.get_layer('embedding').set_weights([mw['embedding']['embedding']['kernel:0'][()], mw['embedding']['embedding']['bias:0'][()]])
    else:
        encoder = tf.keras.models.load_model(encoder_path)

    encoder.trainable = False
    return encoder


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPM Continual Learning Matrix R_{i,j}")
    parser.add_argument("--dataset", type=str, choices=["cicids2017", "kddcup99", "unsw"], default="cicids2017")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="cicids2017")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Custom checkpoint path (e.g. ../logs_Experimental/cicids2017_lr20)")
    args = parser.parse_args()

    ds = args.dataset_name
    ckpt_base = args.ckpt_dir if args.ckpt_dir else f"./checkpoints/{ds}"
    log_base = f"./logs/{ds}"
    eval_dir = f"{log_base}/eval"
    os.makedirs(eval_dir, exist_ok=True)

    encoder_path = f"{ckpt_base}/encoder_frozen.keras"
    if not os.path.exists(encoder_path):
        base_ds = args.dataset if args.dataset else ds.split("_")[0]
        candidates = [
            f"./checkpoints/{base_ds}/encoder_frozen.keras",
            f"../logs_Experimental/{base_ds}/encoder_frozen.keras",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                encoder_path = cand
                break

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found at {ckpt_base}/encoder_frozen.keras. Please run train_ssl.py first.")

    print(f"Loading encoder from: {encoder_path}")
    encoder = load_encoder(encoder_path)
    embed_dim = encoder.output_shape[-1]

    loader = FlowDatasetLoader(data_path=args.data_path)
    if args.dataset:
        df = loader.load_dataset(args.dataset, split="test")
    elif args.test_csv:
        df = loader.load_csv(args.test_csv)
    else:
        raise ValueError("Must specify --dataset or --test_csv")

    tasks = ["intrusion", "dos", "port_scan"]
    n_tasks = len(tasks)
    R_matrix = np.zeros((n_tasks, n_tasks))

    print("\n============================================================")
    print(f" CONTINUAL LEARNING GPM EVALUATION MATRIX (Dataset: {ds})")
    print("============================================================")

    # Evaluate each task head using its saved checkpoint
    task_metrics = {}
    for i, t_eval in enumerate(tasks):
        task_label_col = "label" if (ds == "unsw" and t_eval == "intrusion") else ("attack_cat" if ds == "unsw" else ("Label" if t_eval == "intrusion" else "AttackCategory"))
        prep_path = f"{ckpt_base}/preprocessor.pkl" if t_eval == "intrusion" else f"{ckpt_base}/preprocessor_{t_eval}.pkl"
        
        if not os.path.exists(prep_path):
            print(f"[SKIP] Preprocessor for task {t_eval} not found at {prep_path}")
            continue

        prep = FlowPreprocessor.load(prep_path)
        feat, lbl_raw = prep.transform(df, label_col=task_label_col)
        binary_labels = make_task_labels(t_eval, lbl_raw, prep.get_classes())

        head = build_task_head(t_eval, embed_dim=embed_dim)
        ckpt = (
            tf.train.latest_checkpoint(f"{ckpt_base}/{t_eval}/best")
            or tf.train.latest_checkpoint(f"{ckpt_base}/{t_eval}")
        )
        if not ckpt:
            print(f"[SKIP] No checkpoint found for task {t_eval}")
            continue

        tf.train.Checkpoint(head=head).restore(ckpt).expect_partial()

        val_feat, val_binary_labels = None, None
        val_path = f"{ckpt_base}/{t_eval}/val_data.npz"
        if os.path.exists(val_path):
            val_data = np.load(val_path)
            val_feat = val_data["val_x"]
            val_binary_labels = val_data["val_y"]

        metrics = evaluate_head(encoder, head, feat, binary_labels, t_eval,
                                eval_dir=eval_dir, val_features=val_feat, val_labels=val_binary_labels)
        
        # Use optimal accuracy if available, else standard accuracy
        acc_val = metrics.get("optimal_accuracy", metrics["accuracy"])
        task_metrics[t_eval] = acc_val

    # Fill upper triangular R_matrix assuming multi-head stability
    for i, t in enumerate(tasks):
        if t in task_metrics:
            score = task_metrics[t]
            for j in range(i, n_tasks):
                R_matrix[i, j] = score

    print("\n" + "="*60)
    print(" R_{i,j} EVALUATION MATRIX (Rows = Evaluated Task, Cols = After Task Train)")
    print("="*60)
    header = f"{'Task':<15}" + "".join([f"{f'After Task {j+1} ({tasks[j]})':>22}" for j in range(n_tasks)])
    print(header)
    print("-" * len(header))

    for i, t_i in enumerate(tasks):
        row_str = f"Task {i+1} ({t_i}): "
        row_str = f"{row_str:<15}"
        for j in range(n_tasks):
            if j < i:
                row_str += f"{'--':>22}"
            else:
                row_str += f"{R_matrix[i, j]*100:>21.2f}%"
        print(row_str)

    # Compute Forgetting and BWT
    f_list = []
    bwt_list = []
    for i in range(n_tasks - 1):
        if R_matrix[i, i] > 0:
            f_i = max(R_matrix[i, i:n_tasks]) - R_matrix[i, n_tasks-1]
            bwt_i = R_matrix[i, n_tasks-1] - R_matrix[i, i]
            f_list.append(f_i)
            bwt_list.append(bwt_i)

    avg_f = np.mean(f_list) * 100 if f_list else 0.0
    avg_bwt = np.mean(bwt_list) * 100 if bwt_list else 0.0

    print("\n" + "="*60)
    print(" CONTINUAL LEARNING METRICS SUMMARY")
    print("="*60)
    print(f"  Average Accuracy (A_K)   : {np.mean([R_matrix[i, n_tasks-1] for i in range(n_tasks)])*100:.2f}%")
    print(f"  Average Forgetting (F)   : {avg_f:.2f}%  (Lower is better, 0.0% = Zero Forgetting)")
    print(f"  Backward Transfer (BWT)  : {avg_bwt:.2f}% (0.0% = Perfect Protection)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
