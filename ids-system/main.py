"""
main.py - Master entry point for the SSSL-Based Continual IDS.

Usage:
    python main.py --mode ssl
    python main.py --mode task --task dos
    python main.py --mode evaluate
    python main.py --mode pipeline --train_csv train.csv --test_csv test.csv
    python main.py --mode unsw
    python main.py --mode kddcup99
    python main.py --mode cicids2017
    python main.py --mode predict
    python main.py --mode benchmark
"""
import argparse
import os
import subprocess
import sys


def run(cmd: list[str]):
    print(f"\n>>> {subprocess.list2cmdline(cmd)}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def add_arg(cmd: list[str], name: str, value):
    if value is not None:
        cmd.extend([f"--{name}", str(value)])
    return cmd


def run_csv_pipeline(
    python: str,
    train_csv: str,
    test_csv: str,
    label_col: str,
    task: str,
    ssl_epochs: int,
    task_epochs: int,
    dataset_name: str = "default",
):
    run([
        python, "training/train_ssl.py",
        "--train_csv", train_csv,
        "--label_col", label_col,
        "--epochs", str(ssl_epochs),
        "--dataset_name", dataset_name,
    ])
    run([
        python, "training/train_task.py",
        "--task", task,
        "--train_csv", train_csv,
        "--label_col", label_col,
        "--epochs", str(task_epochs),
        "--dataset_name", dataset_name,
    ])
    run([
        python, "training/evaluate.py",
        "--task", task,
        "--test_csv", test_csv,
        "--label_col", label_col,
        "--dataset_name", dataset_name,
    ])
    run([python, "training/visualize_metrics.py", "--task", task, "--dataset_name", dataset_name])


def run_named_dataset_pipeline(
    python: str,
    dataset: str,
    data_path: str,
    label_col: str,
    task: str,
    ssl_epochs: int,
    task_epochs: int,
    dataset_name: str = "default",
):
    common_args = [
        "--dataset", dataset,
        "--data_path", data_path,
        "--label_col", label_col,
    ]
    run([python, "training/train_ssl.py", *common_args, "--epochs", str(ssl_epochs), "--dataset_name", dataset_name])
    run([python, "training/train_task.py", "--task", task, *common_args, "--epochs", str(task_epochs), "--dataset_name", dataset_name])
    run([python, "training/evaluate.py", "--task", task, *common_args, "--dataset_name", dataset_name])
    run([python, "training/visualize_metrics.py", "--task", task, "--dataset_name", dataset_name])


def main():
    parser = argparse.ArgumentParser(description="SSSL-Based Continual IDS - Master CLI")
    parser.add_argument("--mode", type=str, required=True,
                        choices=[
                            "ssl", "task", "evaluate", "predict", "benchmark",
                            "pipeline", "unsw", "kddcup99", "cicids2017", "anoshift",
                            "visualize", "train_anomaly", "zeroday", "zero_day",
                        ],
                        help="Operating mode")
    parser.add_argument("--task", type=str, default=None, help="Task name (for --mode task/evaluate/pipeline)")
    parser.add_argument("--zeroday_attack", type=str, default=None, help="Target attack category to test as zero-day")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--ssl_epochs", type=int, default=None, help="CSV pipeline SSL epochs")
    parser.add_argument("--task_epochs", type=int, default=None, help="CSV pipeline task epochs")
    parser.add_argument("--train_csv", type=str, default=None, help="Training CSV path")
    parser.add_argument("--test_csv", type=str, default=None, help="Testing CSV path")
    parser.add_argument("--label_col", type=str, default=None, help="Dataset label column")
    parser.add_argument("--dataset", type=str, choices=["cicids2017", "kddcup99", "unsw", "anoshift"],
                        default=None, help="Supported raw dataset for stage-specific modes")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Dataset directory or raw file")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset identifier for scoping output paths (auto-set by dataset modes)")
    args = parser.parse_args()

    python = sys.executable
    dataset_name = args.dataset_name or args.dataset or "default"

    if args.mode == "ssl":
        cmd = [python, "training/train_ssl.py"]
        if args.epochs:
            cmd.extend(["--epochs", str(args.epochs)])
        add_arg(cmd, "train_csv", args.train_csv)
        add_arg(cmd, "dataset", args.dataset)
        add_arg(cmd, "data_path", args.data_path)
        add_arg(cmd, "label_col", args.label_col)
        cmd.extend(["--dataset_name", dataset_name])
        run(cmd)

    elif args.mode == "task":
        if not args.task:
            print("ERROR: --task is required for mode 'task'")
            sys.exit(1)
        cmd = [python, "training/train_task.py", "--task", args.task]
        if args.epochs:
            cmd.extend(["--epochs", str(args.epochs)])
        add_arg(cmd, "train_csv", args.train_csv)
        add_arg(cmd, "dataset", args.dataset)
        add_arg(cmd, "data_path", args.data_path)
        label_col = args.label_col or ("Label" if args.task == "intrusion" else "AttackCategory")
        add_arg(cmd, "label_col", label_col)
        cmd.extend(["--dataset_name", dataset_name])
        run(cmd)

    elif args.mode == "evaluate":
        cmd = [python, "training/evaluate.py"]
        add_arg(cmd, "task", args.task)
        add_arg(cmd, "test_csv", args.test_csv)
        add_arg(cmd, "dataset", args.dataset)
        add_arg(cmd, "data_path", args.data_path)
        label_col = args.label_col or (("Label" if args.task == "intrusion" else "AttackCategory") if args.task else None)
        add_arg(cmd, "label_col", label_col)
        cmd.extend(["--dataset_name", dataset_name])
        run(cmd)

    elif args.mode == "train_anomaly":
        cmd = [python, "training/train_anomaly.py"]
        add_arg(cmd, "dataset", args.dataset)
        data_path = args.data_path or {
            "cicids2017": "../CICIDS2017",
            "kddcup99": "../KDDCUP99",
            "unsw": "../IDS-UNSW_NB",
            "anoshift": "./data/anoshift",
        }.get(args.dataset)
        add_arg(cmd, "data_path", data_path)
        add_arg(cmd, "train_csv", args.train_csv)
        add_arg(cmd, "label_col", args.label_col)
        if args.epochs:
            cmd.extend(["--epochs", str(args.epochs)])
        cmd.extend(["--dataset_name", dataset_name])
        run(cmd)

    elif args.mode in {"zeroday", "zero_day"}:
        cmd = [python, "training/evaluate_zeroday.py"]
        add_arg(cmd, "dataset", args.dataset)
        data_path = args.data_path or {
            "cicids2017": "../CICIDS2017",
            "kddcup99": "../KDDCUP99",
            "unsw": "../IDS-UNSW_NB",
            "anoshift": "./data/anoshift",
        }.get(args.dataset)
        add_arg(cmd, "data_path", data_path)
        add_arg(cmd, "test_csv", args.test_csv)
        add_arg(cmd, "label_col", args.label_col)
        add_arg(cmd, "zeroday_attack", args.zeroday_attack)
        cmd.extend(["--dataset_name", dataset_name])
        run(cmd)

    elif args.mode == "predict":
        run([python, "inference/predict.py", "--dataset_name", dataset_name])

    elif args.mode == "benchmark":
        run([python, "training/benchmark.py"])

    elif args.mode == "visualize":
        task = args.task or "intrusion"
        run([python, "training/visualize_metrics.py", "--task", task, "--dataset_name", dataset_name])

    elif args.mode == "pipeline":
        if not args.train_csv or not args.test_csv:
            print("ERROR: --train_csv and --test_csv are required for mode 'pipeline'")
            sys.exit(1)
        run_csv_pipeline(
            python=python,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            label_col=args.label_col or "Label",
            task=args.task or "intrusion",
            ssl_epochs=args.ssl_epochs or args.epochs or 5,
            task_epochs=args.task_epochs or args.epochs or 5,
            dataset_name=dataset_name,
        )

    elif args.mode == "unsw":
        run_csv_pipeline(
            python=python,
            train_csv=args.train_csv or "../IDS-UNSW_NB/UNSW_NB15_training-set.csv",
            test_csv=args.test_csv or "../IDS-UNSW_NB/UNSW_NB15_testing-set.csv",
            label_col=args.label_col or "label",
            task=args.task or "intrusion",
            ssl_epochs=args.ssl_epochs or args.epochs or 5,
            task_epochs=args.task_epochs or args.epochs or 5,
            dataset_name=args.dataset_name or "unsw",
        )

    elif args.mode in {"kddcup99", "cicids2017", "anoshift"}:
        dataset = args.mode
        data_path = args.data_path or {
            "kddcup99": "../KDDCUP99",
            "cicids2017": "../CICIDS2017",
            "anoshift": "./data/anoshift",
        }[dataset]
        task = args.task or "intrusion"
        label_col = args.label_col or ("Label" if task == "intrusion" else "AttackCategory")
        run_named_dataset_pipeline(
            python=python,
            dataset=dataset,
            data_path=data_path,
            label_col=label_col,
            task=task,
            ssl_epochs=args.ssl_epochs or args.epochs or 5,
            task_epochs=args.task_epochs or args.epochs or 5,
            dataset_name=args.dataset_name or dataset,
        )


if __name__ == "__main__":
    main()