"""
Checkpoint 5b: Evaluate the Unsloth fine-tuned model (200-step checkpoint).

Uses the same 200-test-puzzle test.jsonl as the baseline and mlx_lm eval,
so results are directly comparable across all three.

Run: python 05b_eval_unsloth.py
Produces: unsloth_results.json
"""
import json
import re
from mlx_lm import load, generate

try:
    from mlx_lm.sample_utils import make_sampler
    _SAMPLER = make_sampler(temp=0.0)
except ImportError:
    _SAMPLER = None

# ---- Config ----
MODEL_NAME   = "google/gemma-3-4b-it"
ADAPTER_PATH = "unsloth_adapters/checkpoint-200"
TEST_FILE    = "test.jsonl"
OUTPUT_FILE  = "unsloth_results.json"
# -----------------


def load_test_set(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def parse_grid_from_output(text):
    lines = text.strip().splitlines()
    candidate_rows = []
    for line in lines:
        digits = re.findall(r"\d", line)
        if len(digits) == 9:
            candidate_rows.append([int(d) for d in digits])
    if len(candidate_rows) < 9:
        return None
    return candidate_rows[-9:]


def score_grid(predicted, target):
    if predicted is None:
        return False, 0, False
    correct = sum(
        predicted[r][c] == target[r][c]
        for r in range(9) for c in range(9)
    )
    return True, correct, (correct == 81)


def grid_from_text(text):
    return [[int(v) for v in row.split(",")] for row in text.strip().splitlines()]


def main():
    print(f"Loading model: {MODEL_NAME}")
    print(f"Applying adapters: {ADAPTER_PATH}")
    model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_PATH)

    test_examples = load_test_set(TEST_FILE)
    print(f"Loaded {len(test_examples)} test examples.\n")

    results = []
    total_valid = 0
    total_exact = 0
    total_cell_correct = 0
    total_cells = 0

    for i, example in enumerate(test_examples):
        prompt = example["prompt"]
        target_grid = grid_from_text(example["target"])

        messages = [{"role": "user", "content": prompt}]
        model_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        generate_kwargs = dict(max_tokens=200, verbose=False)
        if _SAMPLER is not None:
            generate_kwargs["sampler"] = _SAMPLER
        else:
            generate_kwargs["temp"] = 0.0

        generated_text = generate(
            model, tokenizer, prompt=model_prompt, **generate_kwargs
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
            n = i + 1
            print(f"  [{n}/{len(test_examples)}] "
                  f"validity={total_valid/n:.2%}  "
                  f"exact={total_exact/n:.2%}  "
                  f"per-cell={total_cell_correct/total_cells:.2%}")

    summary = {
        "model": MODEL_NAME,
        "adapter_path": ADAPTER_PATH,
        "training_steps": 200,
        "n_examples": len(test_examples),
        "validity_rate": total_valid / len(test_examples),
        "exact_match_rate": total_exact / len(test_examples),
        "per_cell_accuracy": total_cell_correct / total_cells,
    }

    print("\n=== UNSLOTH MODEL RESULTS (200 steps) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Three-way comparison
    print("\n=== THREE-WAY COMPARISON ===")
    comparisons = [
        ("baseline_results.json",  "Baseline (no fine-tuning)"),
        ("finetuned_results.json", "mlx_lm LoRA (1000 steps)"),
    ]
    metrics = ["validity_rate", "exact_match_rate", "per_cell_accuracy"]

    rows = {"Unsloth LoRA (200 steps)": summary}
    for path, label in comparisons:
        try:
            with open(path) as f:
                rows[label] = json.load(f)["summary"]
        except FileNotFoundError:
            print(f"  (skipping {label} — {path} not found)")

    print(f"\n  {'Metric':<25}", end="")
    for label in rows:
        print(f"  {label:<28}", end="")
    print()
    print("  " + "-" * 90)
    for m in metrics:
        print(f"  {m:<25}", end="")
        for data in rows.values():
            val = data.get(m, "N/A")
            print(f"  {val:.2%}" if isinstance(val, float) else f"  {val:<28}", end="")
        print()

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)
    print(f"\nSaved full results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()