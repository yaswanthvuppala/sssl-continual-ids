"""
main.py — Master entry point for the SSSL-Based Continual IDS.

Usage:
    python main.py --mode ssl          # Stage 1: SSL Pretraining
    python main.py --mode task --task dos    # Stage 2: Train DoS head
    python main.py --mode task --task port_scan
    python main.py --mode evaluate     # Evaluate all heads
    python main.py --mode pipeline --train_csv train.csv --test_csv test.csv
    python main.py --mode unsw         # Train on UNSW-NB15 train CSV, test on test CSV
    python main.py --mode predict      # Run inference on synthetic data
    python main.py --mode benchmark    # Full end-to-end benchmark
"""
import argparse
import subprocess
import sys
import os


def run(cmd: str):
    print(f"\n>>> {cmd}\n")
    result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def add_arg(cmd: str, name: str, value):
    if value is None:
        return cmd
    return f'{cmd} --{name} "{value}"'


def run_csv_pipeline(
    python: str,
    train_csv: str,
    test_csv: str,
    label_col: str,
    task: str,
    ssl_epochs: int,
    task_epochs: int,
):
    run(
        f'"{python}" training/train_ssl.py --train_csv "{train_csv}" '
        f'--label_col "{label_col}" --epochs {ssl_epochs}'
    )
    run(
        f'"{python}" training/train_task.py --task {task} --train_csv "{train_csv}" '
        f'--label_col "{label_col}" --epochs {task_epochs}'
    )
    run(
        f'"{python}" training/evaluate.py --task {task} --test_csv "{test_csv}" '
        f'--label_col "{label_col}"'
    )
    # Auto-generate visualization plots
    run(f'"{python}" training/visualize_metrics.py --task {task}')



def main():
    parser = argparse.ArgumentParser(description="SSSL-Based Continual IDS — Master CLI")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["ssl", "task", "evaluate", "predict", "benchmark", "pipeline", "unsw", "visualize"],
                        help="Operating mode")
    parser.add_argument("--task", type=str, default=None, help="Task name (for --mode task/evaluate/pipeline)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--ssl_epochs", type=int, default=None, help="CSV pipeline SSL epochs")
    parser.add_argument("--task_epochs", type=int, default=None, help="CSV pipeline task epochs")
    parser.add_argument("--train_csv", type=str, default=None, help="Training CSV path")
    parser.add_argument("--test_csv", type=str, default=None, help="Testing CSV path")
    parser.add_argument("--label_col", type=str, default=None, help="Dataset label column")
    args = parser.parse_args()

    python = sys.executable

    if args.mode == "ssl":
        cmd = f'"{python}" training/train_ssl.py'
        if args.epochs:
            cmd += f" --epochs {args.epochs}"
        cmd = add_arg(cmd, "train_csv", args.train_csv)
        cmd = add_arg(cmd, "label_col", args.label_col)
        run(cmd)

    elif args.mode == "task":
        if not args.task:
            print("ERROR: --task is required for mode 'task'")
            sys.exit(1)
        cmd = f'"{python}" training/train_task.py --task {args.task}'
        if args.epochs:
            cmd += f" --epochs {args.epochs}"
        cmd = add_arg(cmd, "train_csv", args.train_csv)
        cmd = add_arg(cmd, "label_col", args.label_col)
        run(cmd)

    elif args.mode == "evaluate":
        cmd = f'"{python}" training/evaluate.py'
        if args.task:
            cmd += f" --task {args.task}"
        cmd = add_arg(cmd, "test_csv", args.test_csv)
        cmd = add_arg(cmd, "label_col", args.label_col)
        run(cmd)

    elif args.mode == "predict":
        run(f'"{python}" inference/predict.py')

    elif args.mode == "benchmark":
        run(f'"{python}" training/benchmark.py')

    elif args.mode == "visualize":
        task = args.task or "intrusion"
        run(f'"{python}" training/visualize_metrics.py --task {task}')

    elif args.mode == "pipeline":
        if not args.train_csv or not args.test_csv:
            print("ERROR: --train_csv and --test_csv are required for mode 'pipeline'")
            sys.exit(1)
        task = args.task or "intrusion"
        label_col = args.label_col or "Label"
        ssl_epochs = args.ssl_epochs or args.epochs or 5
        task_epochs = args.task_epochs or args.epochs or 5
        run_csv_pipeline(
            python=python,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            label_col=label_col,
            task=task,
            ssl_epochs=ssl_epochs,
            task_epochs=task_epochs,
        )

    elif args.mode == "unsw":
        train_csv = args.train_csv or "../IDS-UNSW_NB/UNSW_NB15_training-set.csv"
        test_csv = args.test_csv or "../IDS-UNSW_NB/UNSW_NB15_testing-set.csv"
        label_col = args.label_col or "label"
        task = args.task or "intrusion"
        ssl_epochs = args.ssl_epochs or args.epochs or 5
        task_epochs = args.task_epochs or args.epochs or 5

        run_csv_pipeline(
            python=python,
            train_csv=train_csv,
            test_csv=test_csv,
            label_col=label_col,
            task=task,
            ssl_epochs=ssl_epochs,
            task_epochs=task_epochs,
        )


if __name__ == "__main__":
    main()
