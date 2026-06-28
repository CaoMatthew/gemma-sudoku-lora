"""
Checkpoint 1: Load sudoku.csv, sample a manageable subset, and create a
train/test split. We FIX the test set here, before any model touches the
data, so baseline and fine-tuned evaluation are guaranteed comparable.

Run: python 01_prepare_data.py
Produces: train.jsonl, test.jsonl  (in this same folder)
"""
import pandas as pd
import json
import random

random.seed(42)  # fixed seed = reproducible split every time you run this

# ---- Config: tweak these if needed ----
CSV_PATH = "sudoku.csv"
N_TRAIN = 5000      # number of puzzles to fine-tune on
N_TEST = 200        # number of puzzles for evaluation (kept small: each eval
                    # run costs one model generation per puzzle)
# ----------------------------------------

INSTRUCTION = (
    "Solve this Sudoku puzzle. Output the completed 9x9 grid, one row per "
    "line, digits only, separated by commas. Do not include any explanation, "
    "just the grid."
)

CLOSING_REMINDER = (
    "\n\nReturn only the completed Sudoku grid. "
    "Use exactly 9 lines. Each line must contain 9 comma-separated digits. "
    "Do not include any explanation."
)


def string_to_grid(s):
    assert len(s) == 81, f"Expected 81 chars, got {len(s)}"
    return [[int(s[r * 9 + c]) for c in range(9)] for r in range(9)]


def grid_to_text(grid):
    return "\n".join(",".join(str(v) for v in row) for row in grid)


def make_prompt(quiz_str):
    grid = string_to_grid(quiz_str)
    return f"{INSTRUCTION}\n\n{grid_to_text(grid)}{CLOSING_REMINDER}"


def make_target(sol_str):
    grid = string_to_grid(sol_str)
    return grid_to_text(grid)


def main():
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows in file: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Basic sanity check on column names — adjust here if your CSV differs
    assert "quizzes" in df.columns and "solutions" in df.columns, (
        "Expected columns 'quizzes' and 'solutions'. "
        f"Found: {list(df.columns)}. Edit this script if your column names differ."
    )

    # Shuffle once, then take what we need. Doing this avoids any ordering
    # bias in the original file (e.g. if it's sorted by difficulty).
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    needed = N_TRAIN + N_TEST
    if len(df) < needed:
        raise ValueError(f"Need {needed} rows but file only has {len(df)}.")

    train_df = df.iloc[:N_TRAIN]
    test_df = df.iloc[N_TRAIN:N_TRAIN + N_TEST]

    # Sanity check: confirm no overlap between train and test puzzle strings
    overlap = set(train_df["quizzes"]) & set(test_df["quizzes"])
    assert len(overlap) == 0, "Data leakage detected between train/test!"
    print("Confirmed: no overlap between train and test sets.")

    def write_jsonl(out_df, path):
        with open(path, "w") as f:
            for _, row in out_df.iterrows():
                example = {
                    "prompt": make_prompt(row["quizzes"]),
                    "target": make_target(row["solutions"]),
                }
                f.write(json.dumps(example) + "\n")
        print(f"Wrote {len(out_df)} examples to {path}")

    write_jsonl(train_df, "train.jsonl")
    write_jsonl(test_df, "test.jsonl")

    # Show one example so you can visually confirm it looks right
    print("\n--- Example training item ---")
    with open("train.jsonl") as f:
        example = json.loads(f.readline())
    print("PROMPT:")
    print(example["prompt"])
    print("\nTARGET:")
    print(example["target"])


if __name__ == "__main__":
    main()