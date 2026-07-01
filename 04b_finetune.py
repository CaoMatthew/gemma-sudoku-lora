"""
Checkpoint 4b: LoRA fine-tuning with mlx_lm.

We use mlx_lm's built-in LoRA trainer, which is the standard, well-maintained
path for fine-tuning MLX models on Apple Silicon. Unsloth is not involved here
(it doesn't add value over native mlx_lm for training on Mac).

What LoRA does:
  Instead of updating all ~4B parameters (expensive, slow, memory-heavy),
  LoRA freezes the base model and inserts small trainable "adapter" matrices
  into the attention layers. Only these adapters are updated during training
  — typically <1% of total parameters. After training, the adapters can be
  merged back into the base model or kept separate.

Run this script with:
    python 04b_finetune.py

Or equivalently, call mlx_lm.lora directly via CLI (same thing):
    python -m mlx_lm.lora \\
        --model mlx-community/gemma-3-text-4b-it-4bit \\
        --train \\
        --data lora_data \\
        --batch-size 1 \\
        --iters 50 \\
        --num-layers 8 \\
        --grad-checkpoint \\
        --mask-prompt \\
        --adapter-path lora_adapters

Produces: lora_adapters/  (directory of trained adapter weights)
"""
import subprocess
import sys

MODEL = "mlx-community/gemma-3-text-4b-it-4bit"
DATA_DIR = "lora_data"
ADAPTER_PATH = "lora_adapters"

# ---- LoRA hyperparameters (tuned for M3 MacBook Air 24GB) ----
BATCH_SIZE = 1          # start at 1 for smoke test; increase to 2 if no OOM
GRAD_ACCUM = 4          # accumulate gradients over 4 steps -> effective batch 4/8
ITERS = 50              # smoke test: confirm training runs before committing to 1000
NUM_LAYERS = 8          # number of transformer layers to attach LoRA adapters to
                        # (counts from the last layer inward — later layers
                        # tend to be most task-relevant for structured outputs)
LEARNING_RATE = 2e-4    # standard LoRA learning rate; safe starting point
# ---------------------------------------------------------------


def main():
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", MODEL,
        "--train",
        "--data", DATA_DIR,
        "--adapter-path", ADAPTER_PATH,
        "--batch-size", str(BATCH_SIZE),
        "--grad-accumulation-steps", str(GRAD_ACCUM),
        "--iters", str(ITERS),
        "--num-layers", str(NUM_LAYERS),
        "--learning-rate", str(LEARNING_RATE),
        "--grad-checkpoint",
        "--mask-prompt",  # only compute loss on the assistant completion (the grid),
                          # not on the puzzle input — this is the part we care about
        "--save-every", "50", #CHANGE BEFORE ACTUAL RUN!!! save adapters every N iterations (useful for long runs)
        "--val-batches", "25",  # how many validation batches to evaluate per checkpoint
    ]

    print("Starting LoRA fine-tuning...")
    print("Command:", " ".join(cmd))
    print("-" * 60)
    print("What to watch for:")
    print("  'Train loss' should decrease over iterations")
    print("  'Val loss' should also decrease (if it starts rising, overfitting)")
    print("  Memory warnings or OOM = reduce --batch-size to 1")
    print("-" * 60)

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\nTraining complete. Adapters saved to: {ADAPTER_PATH}/")
        print("Next step: run 05_eval_finetuned.py to evaluate the fine-tuned model")
    else:
        print("\nTraining failed. Common fixes:")
        print("  - OOM error: reduce BATCH_SIZE to 1 in this script")
        print("  - Arg not recognized: your mlx_lm version may differ;")
        print("    run `python -m mlx_lm.lora --help` to check available args")


if __name__ == "__main__":
    main()