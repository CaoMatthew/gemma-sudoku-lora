# Gemma Sudoku Fine-Tuning

Fine-tuning and evaluating Gemma 3 4B on Sudoku completion using LoRA on Apple Silicon.

This project tests whether a small LoRA fine-tune can improve a pretrained Gemma model’s ability to output completed Sudoku grids in a strict 9x9 format.

Overview

The pipeline:

Prepare Sudoku train/test data from sudoku.csv
Evaluate the pretrained baseline model
Convert data into chat format for LoRA training
Fine-tune with MLX-LM LoRA on Mac
Fine-tune with Unsloth MLX on Mac
Evaluate the fine-tuned models on the same fixed test set
Compare baseline, MLX-LM LoRA, and Unsloth LoRA results

The main goal is a fair comparison: every model is evaluated on the same test.jsonl file using the same parser and scoring metrics.

Project Structure

Data Preparation

Start with a sudoku.csv file containing two columns:

quizzes
solutions

Each puzzle and solution should be an 81-character string.

Run:

python 01_prepare_data.py

This produces:

train.jsonl
test.jsonl

The script uses a fixed random seed so the train/test split is reproducible.

Current split:

Training examples: 5,000
Test examples: 200

The test set is fixed before any model training happens, which makes baseline and fine-tuned results directly comparable.

Baseline Evaluation

Run:

python 03_baseline_eval.py

This evaluates the pretrained model without fine-tuning:

mlx-community/gemma-3-text-4b-it-4bit

It produces:

baseline_results.json

The baseline is the control group.

Convert Data for LoRA Training

Run:

python 04a_convert_data.py

This converts the original prompt / target JSONL format into chat-style messages format for training.

Input:

train.jsonl

Output:

lora_data/train.jsonl
lora_data/valid.jsonl

The validation split is taken only from the training data. The final test.jsonl remains untouched.

MLX-LM LoRA Fine-Tuning

Run:

python 04b_finetune_mlx_lm.py

This uses MLX-LM’s built-in LoRA trainer on Apple Silicon.

It fine-tunes:

mlx-community/gemma-3-text-4b-it-4bit

using the converted training data in:

lora_data/

Key training choices:

Batch size: 2
Gradient accumulation: 4
Effective batch size: 8
Training iterations: 1000
LoRA layers: 8
Learning rate: 2e-4
Prompt masking: enabled

The script:

Loads the MLX-compatible Gemma 3 4B model
Uses lora_data/train.jsonl for training
Uses lora_data/valid.jsonl for validation
Applies LoRA adapters to selected model layers
Masks prompt tokens so loss is computed only on the assistant Sudoku grid
Saves adapters to mlx_lm_lora_adapters/
Evaluate MLX-LM Fine-Tuned Model

Run:

python 05_eval_finetuned.py

This loads the same base model as the baseline, but applies the MLX-LM LoRA adapters before evaluation.

It evaluates on:

test.jsonl

and produces:

finetuned_results.json

This result is directly comparable to baseline_results.json because it uses the same test set, parser, generation settings, and scoring metrics.

Test Unsloth MLX on Mac

Before a full Unsloth training run, use the smoke test:

python 04c_test_unsloth_mac.py

This checks whether Unsloth’s MLX training path works on the local machine.

It creates:

unsloth_mlx_smoke_adapters/

This folder is generated output and should not be committed.

Unsloth MLX Fine-Tuning

Run:

python 04c_finetune_unsloth.py

This uses:

google/gemma-3-4b-it

with Unsloth’s MLX stack.

Key training choices:

LoRA rank: 8
LoRA alpha: 16
LoRA dropout: 0.0
Batch size: 2
Gradient accumulation: 4
Learning rate: 2e-4
Max sequence length: 512

The script:

Loads the original Hugging Face Gemma checkpoint
Applies runtime 4-bit quantization
Attaches LoRA adapters
Applies the model chat template to training examples
Masks prompt tokens so loss is computed only on the assistant Sudoku grid
Trains using Unsloth’s MLXTrainer
Saves adapters to unsloth_adapters/

The adapter folder is ignored by Git because it contains generated model weights.

Evaluate Unsloth Fine-Tuned Model

Run:

python 05b_unsloth_eval.py

This evaluates the Unsloth adapter checkpoint on the same test.jsonl used by the baseline and MLX-LM evaluation.

It produces:

unsloth_results.json

The evaluator tracks:

validity_rate
exact_match_rate
per_cell_accuracy
Metrics
Validity Rate

The percentage of model outputs that can be parsed as a 9x9 grid.

This measures formatting compliance.

Exact Match Rate

The percentage of puzzles where all 81 cells match the ground-truth solution.

This is the strict Sudoku success metric.

Per-Cell Accuracy

The percentage of individual cells that match the target solution.

This is useful for measuring partial improvement even when exact puzzle solves remain low.

Current Results

Latest comparison:

Model	Validity Rate	Exact Match Rate	Per-Cell Accuracy
Baseline, no fine-tuning	100.00%	0.00%	45.43%
MLX-LM LoRA, 1000 steps	98.50%	0.00%	58.37%
Unsloth LoRA, 200 steps	100.00%	0.00%	65.72%

The Unsloth LoRA run improved per-cell accuracy over both the baseline and the MLX-LM LoRA run, but exact Sudoku solves remain at 0%.