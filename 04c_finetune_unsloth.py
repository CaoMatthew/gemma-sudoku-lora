"""
Checkpoint 4c: LoRA fine-tuning with Unsloth MLX (FastModel API).

Uses Unsloth's native MLX training stack — FastModel, MLXTrainer,
MLXTrainingConfig — which works on Apple Silicon when loaded via the
original HuggingFace checkpoint (runtime-quantized by Unsloth) rather
than a pre-quantized mlx-community build.

Key difference from 04b_finetune_mlxlm.py:
  - Unsloth does runtime 4-bit affine quantization at load time
  - LoRA targets are set explicitly via get_peft_model() in Python
  - Training uses Unsloth's MLXTrainer (CCE loss, gradient checkpointing,
    Metal memory guard) rather than the mlx_lm.lora CLI

Run: python 04c_finetune_unsloth.py
Produces: unsloth_adapters/  (compare against lora_adapters/ from 04b)

Note on model download: google/gemma-3-4b-it is ~8.6GB (full precision,
quantized at runtime). Subsequent runs use the cached copy.
"""
import inspect
import json
import os
from pathlib import Path
from datasets import Dataset
from unsloth import FastModel
from unsloth_zoo.mlx.trainer import (
    MLXTrainer,
    MLXTrainingConfig,
    train_on_responses_only,
)

# ---- Config ----
MODEL_NAME   = "google/gemma-3-4b-it"    # original HF checkpoint; Unsloth
                                          # quantizes at runtime (4-bit affine)
ADAPTER_PATH = "unsloth_adapters"         # separate from lora_adapters/ (04b)
DATA_DIR     = Path("lora_data")          # same data as 04b — fair comparison
MAX_SEQ_LEN  = 512
# ----------------

# ---- LoRA hyperparameters tuned for Apple Silicon ----
LORA_RANK    = 8
LORA_ALPHA   = 16       # effective LR multiplier = alpha / rank = 2.0
LORA_DROPOUT = 0.0
# ------------------------------------------------------

# ---- Training hyperparameters ----
ITERS         = 1000    # 1000 steps × batch 2 ≈ 2000 examples (~0.4 epochs
                        # over ~4750 train examples)
BATCH_SIZE    = 2       # peak mem ~4.9GB at batch 1; batch 2 safe on 24GB
GRAD_ACCUM    = 4       # effective batch = BATCH_SIZE × GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
# ----------------------------------


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_config(config_cls, kwargs):
    """
    MLXTrainingConfig may not expose every TRL-style kwarg.
    Drop unsupported keys gracefully rather than crashing.
    """
    sig = inspect.signature(config_cls)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return config_cls(**kwargs)
    allowed  = {k: v for k, v in kwargs.items() if k in params}
    skipped  = {k: v for k, v in kwargs.items() if k not in params}
    if skipped:
        print(f"  (Skipping unsupported MLXTrainingConfig args: {list(skipped)})")
    return config_cls(**allowed)


def main():
    # ---- Step 1: Load model ----
    print(f"Loading model: {MODEL_NAME}")
    print("(First run downloads ~8.6GB; subsequent runs use cache)")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        full_finetuning=False,
        text_only=True,     # for Gemma 3 4B, Unsloth still uses mlx-vlm wrapper,
                            # but trains the text path/tokenizer for this text-only task
    )
    print(f"Model loaded. Type: {type(model).__name__}")

    # ---- Step 2: Attach LoRA adapters ----
    print("Attaching LoRA adapters...")
    model = FastModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        finetune_vision_layers=False,   # text-only task, no vision layers
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=False,     # attention-only LoRA; more focused
        use_gradient_checkpointing=True,
        random_state=42,
    )

    # ---- Step 3: Prepare dataset ----
    print(f"Loading data from {DATA_DIR}/")
    raw_train = load_jsonl(DATA_DIR / "train.jsonl")
    raw_valid = load_jsonl(DATA_DIR / "valid.jsonl")

    def to_text(examples):
        return [
            tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            for ex in examples
        ]

    train_dataset = Dataset.from_list([{"text": t} for t in to_text(raw_train)])
    valid_dataset = Dataset.from_list([{"text": t} for t in to_text(raw_valid)])
    print(f"Train: {len(train_dataset)} examples, Valid: {len(valid_dataset)} examples")

    # ---- Step 4: Train ----
    print("Configuring MLXTrainer...")
    training_config = build_config(MLXTrainingConfig, dict(
        output_dir=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=ITERS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=200,
        eval_steps=100,     # validation loss every 100 steps
        report_to="none",
        dataset_text_field="text",
    ))

    # MLXTrainer accepts either 'tokenizer' or 'processing_class' depending
    # on the Unsloth version — detect which one to pass
    trainer_sig = inspect.signature(MLXTrainer)
    tokenizer_kwarg = (
        "tokenizer" if "tokenizer" in trainer_sig.parameters
        else "processing_class"
    )

    trainer = MLXTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,   # validation loss logged every eval_steps
        args=training_config,
        **{tokenizer_kwarg: tokenizer},
    )

    # Mask the user prompt tokens so loss is computed only on the assistant
    # grid response — equivalent to --mask-prompt in the mlx_lm run (04b),
    # ensures both fine-tuning approaches are comparable.
    # Gemma's chat template uses <start_of_turn>user/model as turn markers.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    print("-" * 60)
    print("Starting training...")
    print("Watch for: loss decreasing, no OOM, Peak mem staying under ~20GB")
    print("-" * 60)
    trainer.train()

    # ---- Step 5: Save adapters ----
    print("Saving adapters...")
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(ADAPTER_PATH)
    else:
        trainer.save_model(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)

    print(f"\nTraining complete. Adapters saved to: {ADAPTER_PATH}/")
    print("Next: run 05_eval_finetuned.py (update ADAPTER_PATH to 'unsloth_adapters')")


if __name__ == "__main__":
    main()