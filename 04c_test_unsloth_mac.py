"""
Checkpoint 4c: Unsloth MLX Mac smoke test.

Purpose:
    Test whether Unsloth can run LoRA training locally on Apple Silicon / MLX.

Run:
    python 04c_test_unsloth_mlx_mac.py

Then try:
    UNSLOTH_MODEL=google/gemma-3-4b-it UNSLOTH_TEXT_ONLY=1 python 04c_test_unsloth_mlx_mac.py
"""

import inspect
import json
import os
import platform
import traceback
from pathlib import Path


# Start with Gemma 3 1B because it is text-only.
# Gemma 3 4B/12B/27B are multimodal, so 4B adds VLM routing complexity.
MODEL_NAME = os.environ.get("UNSLOTH_MODEL", "google/gemma-3-1b-it")
TEXT_ONLY = os.environ.get("UNSLOTH_TEXT_ONLY", "1") == "1"

DATA_DIR = Path("lora_data")
OUTPUT_DIR = os.environ.get("UNSLOTH_OUTPUT_DIR", "unsloth_mlx_smoke_adapters")

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "1024"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "2"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_tiny_examples(path, n=8):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= n:
                break
    return examples


def build_config(config_cls, desired_kwargs):
    """
    Keep this tolerant because MLXTrainingConfig may not expose exactly
    the same arguments as TRL's SFTConfig.
    """
    sig = inspect.signature(config_cls)
    params = sig.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return config_cls(**desired_kwargs)

    allowed = {}
    skipped = {}

    for k, v in desired_kwargs.items():
        if k in params:
            allowed[k] = v
        else:
            skipped[k] = v

    if skipped:
        print("Skipping unsupported MLXTrainingConfig args:")
        for k in skipped:
            print(f"  - {k}")

    return config_cls(**allowed)


def main():
    print_section("System info")
    print(f"Platform:  {platform.platform()}")
    print(f"Python:    {platform.python_version()}")
    print(f"Model:     {MODEL_NAME}")
    print(f"Text only: {TEXT_ONLY}")
    print(f"Output:    {OUTPUT_DIR}")

    print_section("Step 1: Import Unsloth MLX stack")
    try:
        import unsloth
        from unsloth import FastModel
        from unsloth_zoo.mlx.loader import FastMLXModel
        from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
        from datasets import Dataset

        print(f"Unsloth version: {getattr(unsloth, '__version__', 'unknown')}")
        print(f"Unsloth DEVICE_TYPE: {getattr(unsloth, 'DEVICE_TYPE', 'unknown')}")
        print("Imported FastModel, FastMLXModel, MLXTrainer, MLXTrainingConfig, Dataset.")
    except Exception as e:
        print("FAILED importing Unsloth MLX stack.")
        print(type(e).__name__ + ":", e)
        traceback.print_exc()
        return

    print_section("Step 2: Load model through Unsloth MLX")
    try:
        load_kwargs = dict(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LEN,
            load_in_4bit=True,
            full_finetuning=False,
        )

        if TEXT_ONLY:
            load_kwargs["text_only"] = True

        try:
            model, tokenizer = FastModel.from_pretrained(**load_kwargs)
        except TypeError as e:
            if "text_only" in str(e):
                print("This FastModel.from_pretrained does not accept text_only; retrying without it.")
                load_kwargs.pop("text_only", None)
                model, tokenizer = FastModel.from_pretrained(**load_kwargs)
            else:
                raise

        print("Model loaded successfully.")
        print(f"Model type:     {type(model).__name__}")
        print(f"Tokenizer type: {type(tokenizer).__name__}")

    except Exception as e:
        print("FAILED during Unsloth MLX model loading.")
        print(type(e).__name__ + ":", e)
        traceback.print_exc()
        return

    print_section("Step 3: Attach LoRA adapters")
    try:
        model = FastModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=False,
            use_gradient_checkpointing=True,
            random_state=42,
        )
        print("LoRA adapters attached successfully.")
        print(f"LoRA model type: {type(model).__name__}")

    except Exception as e:
        print("FAILED while attaching LoRA adapters.")
        print(type(e).__name__ + ":", e)
        traceback.print_exc()
        return

    print_section("Step 4: Build tiny training dataset")
    try:
        train_path = DATA_DIR / "train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(f"Could not find {train_path}")

        raw_examples = load_tiny_examples(train_path, n=8)

        texts = []
        for ex in raw_examples:
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append({"text": text})

        train_dataset = Dataset.from_list(texts)

        print(f"Built tiny dataset with {len(train_dataset)} examples.")
        print("\nExample text preview:")
        print(train_dataset[0]["text"][:500])

    except Exception as e:
        print("FAILED preparing tiny dataset.")
        print(type(e).__name__ + ":", e)
        traceback.print_exc()
        return

    print_section("Step 5: Run 2-step Unsloth MLX training smoke test")
    try:
        desired_config = dict(
            output_dir=OUTPUT_DIR,
            max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            logging_steps=1,
            save_steps=MAX_STEPS,
            report_to="none",
            dataset_text_field="text",
        )

        args = build_config(MLXTrainingConfig, desired_config)

        trainer_sig = inspect.signature(MLXTrainer)
        trainer_kwargs = dict(
            model=model,
            train_dataset=train_dataset,
            args=args,
        )

        if "tokenizer" in trainer_sig.parameters:
            trainer_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in trainer_sig.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            print("MLXTrainer signature does not expose tokenizer or processing_class.")
            print("Trainer signature:")
            print(trainer_sig)

        trainer = MLXTrainer(**trainer_kwargs)

        trainer.train()

        print("\nTraining finished. Saving model/tokenizer...")

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(OUTPUT_DIR)
        elif hasattr(trainer, "save_model"):
            trainer.save_model(OUTPUT_DIR)
        else:
            print("Warning: neither model.save_pretrained nor trainer.save_model exists.")

        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(OUTPUT_DIR)

        print("\nSUCCESS: Unsloth MLX completed a tiny local Mac LoRA training smoke test.")
        print(f"Saved smoke-test adapters to: {OUTPUT_DIR}/")

    except Exception as e:
        print("FAILED during Unsloth MLX training.")
        print(type(e).__name__ + ":", e)
        print("\nMLXTrainer signature:")
        try:
            print(inspect.signature(MLXTrainer))
        except Exception:
            pass
        print("\nMLXTrainingConfig signature:")
        try:
            print(inspect.signature(MLXTrainingConfig))
        except Exception:
            pass
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()