import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

"""
python analyze_training_log.py log_position/train_ablation2_NoPE.jsonl \
  --batch_size 256 \
  --context_length 256 \
  --smooth_window 3 \
  --plot_extra
"""


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def moving_average(values: list[float], window_size: int) -> list[float]:
    if window_size <= 1:
        return values

    result = []
    running_sum = 0.0

    for i, value in enumerate(values):
        running_sum += value

        if i >= window_size:
            running_sum -= values[i - window_size]
            result.append(running_sum / window_size)
        else:
            result.append(running_sum / (i + 1))

    return result


def load_records(log_path: str):
    train_records = []
    val_records = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warning] skip invalid json at line {line_no}")
                continue

            if "step" not in obj:
                continue

            step = int(obj["step"])

            if "loss" in obj:
                train_records.append({
                    "step": step,
                    "loss": float(obj["loss"]),
                    "lr": float(obj["lr"]) if "lr" in obj else None,
                    "grad_norm": float(obj["grad_norm"]) if "grad_norm" in obj else None,
                    "step_time": float(obj["step_time"]) if "step_time" in obj else None,
                })

            if "val_loss" in obj:
                val_records.append({
                    "step": step,
                    "val_loss": float(obj["val_loss"]),
                })

    train_records.sort(key=lambda x: x["step"])
    val_records.sort(key=lambda x: x["step"])

    return train_records, val_records


def estimate_training_time_interval_avg(train_records, start_step: int = 0):
    """
    假设每条 train log 的 step_time 表示：
    从上一个 log step 到当前 step 这段区间的平均单步耗时。

    例如：
        {"step": 100, "step_time": 0.205}
    表示 step 1~100 平均每步 0.205 秒。
    """
    total_seconds = 0.0
    total_steps = 0
    prev_step = start_step

    for r in train_records:
        step = r["step"]
        step_time = r["step_time"]

        if step_time is None:
            continue

        if step <= prev_step:
            continue

        delta_steps = step - prev_step
        interval_seconds = delta_steps * step_time

        total_seconds += interval_seconds
        total_steps += delta_steps
        prev_step = step

    avg_step_time = total_seconds / total_steps if total_steps > 0 else 0.0
    return total_seconds, total_steps, avg_step_time


def estimate_training_time_point_sample(train_records):
    """
    如果 step_time 只是当前单个 step 的耗时，而不是区间平均耗时，
    那么只能用所有记录的平均 step_time 估算总训练时间。
    """
    step_times = [
        r["step_time"]
        for r in train_records
        if r["step_time"] is not None
    ]

    if not step_times or not train_records:
        return 0.0, 0, 0.0

    avg_step_time = sum(step_times) / len(step_times)
    total_steps = max(r["step"] for r in train_records)
    total_seconds = avg_step_time * total_steps

    return total_seconds, total_steps, avg_step_time


def plot_loss_curve(
    train_records,
    val_records,
    output_path: Path,
    smooth_window: int = 1,
):
    train_steps = [r["step"] for r in train_records]
    train_losses = [r["loss"] for r in train_records]

    val_steps = [r["step"] for r in val_records]
    val_losses = [r["val_loss"] for r in val_records]

    if smooth_window > 1:
        train_losses_to_plot = moving_average(train_losses, smooth_window)
        train_label = f"train loss, moving avg window={smooth_window}"
    else:
        train_losses_to_plot = train_losses
        train_label = "train loss"

    plt.figure(figsize=(10, 6))

    if train_steps:
        plt.plot(train_steps, train_losses_to_plot, label=train_label)

    if val_steps:
        plt.plot(val_steps, val_losses, marker="o", label="validation loss")

    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_lr_curve(train_records, output_path: Path):
    records = [r for r in train_records if r["lr"] is not None]

    if not records:
        return False

    steps = [r["step"] for r in records]
    lrs = [r["lr"] for r in records]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs)
    plt.xlabel("step")
    plt.ylabel("learning rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return True


def plot_grad_norm_curve(train_records, output_path: Path, smooth_window: int = 1):
    records = [r for r in train_records if r["grad_norm"] is not None]

    if not records:
        return False

    steps = [r["step"] for r in records]
    grad_norms = [r["grad_norm"] for r in records]

    if smooth_window > 1:
        grad_norms = moving_average(grad_norms, smooth_window)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, grad_norms)
    plt.xlabel("step")
    plt.ylabel("gradient norm")
    plt.title("Gradient Norm")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return True


def print_summary(
    train_records,
    val_records,
    total_seconds: float,
    total_steps: int,
    avg_step_time: float,
    batch_size: int | None,
    context_length: int | None,
):
    print("===== Training Log Summary =====")
    print(f"Number of train records : {len(train_records)}")
    print(f"Number of val records   : {len(val_records)}")

    if train_records:
        print(f"First train step        : {train_records[0]['step']}")
        print(f"Last train step         : {train_records[-1]['step']}")
        print(f"First train loss        : {train_records[0]['loss']:.6f}")
        print(f"Last train loss         : {train_records[-1]['loss']:.6f}")

    if val_records:
        best_val = min(val_records, key=lambda x: x["val_loss"])
        print(f"First val step          : {val_records[0]['step']}")
        print(f"Last val step           : {val_records[-1]['step']}")
        print(f"First val loss          : {val_records[0]['val_loss']:.6f}")
        print(f"Last val loss           : {val_records[-1]['val_loss']:.6f}")
        print(f"Best val loss           : {best_val['val_loss']:.6f} at step {best_val['step']}")

    print()
    print("===== Time Estimate =====")
    print(f"Estimated trained steps : {total_steps}")
    print(f"Estimated total time    : {format_seconds(total_seconds)}")
    print(f"Estimated total seconds : {total_seconds:.2f}")
    print(f"Average step time       : {avg_step_time:.6f}s / step")

    if batch_size is not None and context_length is not None and total_seconds > 0:
        tokens_per_step = batch_size * context_length
        total_tokens = tokens_per_step * total_steps
        tokens_per_second = total_tokens / total_seconds

        print()
        print("===== Throughput Estimate =====")
        print(f"Batch size              : {batch_size}")
        print(f"Context length          : {context_length}")
        print(f"Tokens per step         : {tokens_per_step:,}")
        print(f"Estimated total tokens  : {total_tokens:,}")
        print(f"Estimated throughput    : {tokens_per_second:,.2f} tokens/s")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "log_path",
        type=str,
        help="Path to train jsonl log file",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save plots. Default: same directory as log file.",
    )

    parser.add_argument(
        "--start_step",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--context_length",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
        help="Moving average window for train loss and grad norm.",
    )

    parser.add_argument(
        "--time_mode",
        type=str,
        default="interval_avg",
        choices=["interval_avg", "point_sample"],
        help=(
            "interval_avg: step_time means average time per step since previous log; "
            "point_sample: step_time means sampled single-step time."
        ),
    )

    parser.add_argument(
        "--plot_extra",
        action="store_true",
        help="Also plot learning rate and gradient norm curves.",
    )

    args = parser.parse_args()

    log_path = Path(args.log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")

    if args.out_dir is None:
        out_dir = log_path.parent
    else:
        out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    train_records, val_records = load_records(str(log_path))

    if not train_records and not val_records:
        print("No valid train or val records found.")
        return

    if args.time_mode == "interval_avg":
        total_seconds, total_steps, avg_step_time = estimate_training_time_interval_avg(
            train_records,
            start_step=args.start_step,
        )
    else:
        total_seconds, total_steps, avg_step_time = estimate_training_time_point_sample(
            train_records
        )

    print_summary(
        train_records=train_records,
        val_records=val_records,
        total_seconds=total_seconds,
        total_steps=total_steps,
        avg_step_time=avg_step_time,
        batch_size=args.batch_size,
        context_length=args.context_length,
    )

    stem = log_path.stem

    loss_plot_path = out_dir / f"{stem}_loss_curve.png"
    plot_loss_curve(
        train_records=train_records,
        val_records=val_records,
        output_path=loss_plot_path,
        smooth_window=args.smooth_window,
    )
    print()
    print(f"Saved loss curve        : {loss_plot_path}")

    if args.plot_extra:
        lr_plot_path = out_dir / f"{stem}_lr_curve.png"
        grad_plot_path = out_dir / f"{stem}_grad_norm_curve.png"

        if plot_lr_curve(train_records, lr_plot_path):
            print(f"Saved lr curve          : {lr_plot_path}")

        if plot_grad_norm_curve(
            train_records,
            grad_plot_path,
            smooth_window=args.smooth_window,
        ):
            print(f"Saved grad norm curve   : {grad_plot_path}")


if __name__ == "__main__":
    main()