"""
Checkpoint 4a: Convert train.jsonl into the format mlx_lm.lora expects.

mlx_lm.lora supports a {"messages": [...]} chat format that applies the
model's own chat template during training — this is what we want, because it
means training and inference see exactly the same token layout.

Run: python 04a_convert_data.py
Produces: lora_data/train.jsonl  (and lora_data/valid.jsonl from a small split)
"""
import json
import os
import random

random.seed(42)

TRAIN_FILE = "train.jsonl"
OUT_DIR = "lora_data"
VALID_SPLIT = 0.05  # hold out 5% of training data as validation for mlx_lm

os.makedirs(OUT_DIR, exist_ok=True)


def convert_example(example):
    """
    Convert from our {"prompt": ..., "target": ...} format to mlx_lm's
    {"messages": [...]} chat format.

    The "assistant" message is the exact target grid we want the model to
    produce — this is what the loss is computed on during training.
    """
    return {
        "messages": [
            {"role": "user",    "content": example["prompt"]},
            {"role": "assistant", "content": example["target"]},
        ]
    }


def main():
    with open(TRAIN_FILE) as f:
        examples = [json.loads(line) for line in f]

    random.shuffle(examples)
    n_valid = max(1, int(len(examples) * VALID_SPLIT))
    valid_examples = examples[:n_valid]
    train_examples = examples[n_valid:]

    def write(data, path):
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(convert_example(ex)) + "\n")
        print(f"Wrote {len(data)} examples to {path}")

    write(train_examples, os.path.join(OUT_DIR, "train.jsonl"))
    write(valid_examples, os.path.join(OUT_DIR, "valid.jsonl"))

    # Show one converted example so you can sanity check it
    print("\n--- Example converted training item ---")
    print(json.dumps(convert_example(train_examples[0]), indent=2))


if __name__ == "__main__":
    main()