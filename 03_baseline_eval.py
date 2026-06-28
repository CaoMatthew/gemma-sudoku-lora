"""
Checkpoint 3: Baseline evaluation.
Loads the PRETRAINED (no fine-tuning) Gemma 4B model and evaluates it on
test.jsonl. This is the control group we compare the fine-tuned model against.

Run: python 03_baseline_eval.py
Produces: baseline_results.json
"""
import json
import re
from unsloth import FastLanguageModel

# ---- Config ----
MODEL_NAME = "mlx-community/gemma-3-4b-it-4bit"  # 4-bit instruction-tuned Gemma 3, MLX build
MAX_SEQ_LENGTH = 1024
TEST_FILE = "test.jsonl"
OUTPUT_FILE = "baseline_results.json"
# -----------------


def load_test_set(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def parse_grid_from_output(text):
    """
    Robustly extract a 9x9 grid of digits from raw model output.
    Models often add stray text/code fences even when told not to, so we
    search for any 9 lines that each contain exactly 9 digits (comma or
    space separated) rather than assuming the output is clean.
    """
    lines = text.strip().splitlines()
    candidate_rows = []
    for line in lines:
        digits = re.findall(r"\d", line)
        if len(digits) == 9:
            candidate_rows.append([int(d) for d in digits])
    if len(candidate_rows) < 9:
        return None  # not enough valid rows found -> invalid output
    # Take the last 9 valid rows found (in case of preamble before the grid)
    grid = candidate_rows[-9:]
    return grid


def score_grid(predicted, target):
    """
    Returns (is_valid, per_cell_correct_count, is_exact_match)
    predicted: 9x9 grid or None
    target: 9x9 grid (ground truth)
    """
    if predicted is None:
        return False, 0, False
    correct = 0
    for r in range(9):
        for c in range(9):
            if predicted[r][c] == target[r][c]:
                correct += 1
    is_exact = (correct == 81)
    return True, correct, is_exact


def grid_from_text(text):
    return [[int(v) for v in row.split(",")] for row in text.strip().splitlines()]


def main():
    print(f"Loading model: {MODEL_NAME} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # enables faster inference mode

    test_examples = load_test_set(TEST_FILE)
    print(f"Loaded {len(test_examples)} test examples.")

    results = []
    total_valid = 0
    total_exact = 0
    total_cell_correct = 0
    total_cells = 0

    for i, example in enumerate(test_examples):
        prompt = example["prompt"]
        target_grid = grid_from_text(example["target"])

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )

        outputs = model.generate(
            inputs,
            max_new_tokens=200,   # 81 digits + commas + newlines fits comfortably
            temperature=0.1,      # low temp: we want the model's best deterministic guess
            do_sample=False,
        )
        generated_text = tokenizer.decode(
            outputs[0][inputs.shape[-1]:], skip_special_tokens=True
        )

        predicted_grid = parse_grid_from_output(generated_text)
        is_valid, cell_correct, is_exact = score_grid(predicted_grid, target_grid)

        total_valid += int(is_valid)
        total_exact += int(is_exact)
        total_cell_correct += cell_correct
        total_cells += 81

        results.append({
            "index": i,
            "raw_output": generated_text,
            "is_valid": is_valid,
            "cell_correct": cell_correct,
            "is_exact_match": is_exact,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(test_examples)}] "
                  f"running validity={total_valid/(i+1):.2%} "
                  f"exact={total_exact/(i+1):.2%} "
                  f"per-cell={total_cell_correct/total_cells:.2%}")

    summary = {
        "n_examples": len(test_examples),
        "validity_rate": total_valid / len(test_examples),
        "exact_match_rate": total_exact / len(test_examples),
        "per_cell_accuracy": total_cell_correct / total_cells,
    }

    print("\n=== BASELINE RESULTS ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)
    print(f"\nSaved full results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()