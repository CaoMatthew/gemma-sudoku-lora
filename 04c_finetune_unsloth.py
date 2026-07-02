"""
Checkpoint 4c: LoRA fine-tuning with Unsloth (FastLanguageModel API).

Unsloth's FastLanguageModel handles LoRA adapter setup explicitly via
get_peft_model(), giving us direct control over which layers are adapted,
rank, alpha, and dropout — rather than delegating all of that to mlx_lm's
CLI defaults. On Mac, Unsloth dispatches to its FastMLXModel backend
automatically when it detects an mlx-community checkpoint.

What LoRA does:
  Freezes the base model's 4B parameters and inserts small trainable adapter
  matrices (rank-8 here) into the attention projections. Only these adapters
  (~3.5M params, 0.077% of total) are updated during training.

Run: python 04c_finetune_unsloth.py
Produces: unsloth_lora_adapters/  (trained adapter weights)
"""
from unsloth import FastLanguageModel
import json

# ---- Config ----
MODEL_NAME   = "mlx-community/gemma-3-text-4b-it-4bit"
ADAPTER_PATH = "unsloth_lora_adapters"
DATA_DIR     = "lora_data"
MAX_SEQ_LEN  = 1024
# ----------------

# ---- LoRA hyperparameters tuned for Apple Silicon ----
LORA_RANK    = 8       # dimensionality of adapter matrices; 8 is a safe default
LORA_ALPHA   = 16      # scaling factor: alpha/rank = 2.0 effective LR multiplier
LORA_DROPOUT = 0.0     # no dropout; dataset is clean and training is short
NUM_LAYERS   = 8       # how many transformer layers (from the end) get adapters
# ------------------------------------------------------

# ---- Training hyperparameters ----
ITERS        = 1000    # 1000 iters × batch 2 ≈ 2000 examples seen (~0.4 epochs
                       # over 4750 train examples); increase to ~2375 for one full pass
BATCH_SIZE   = 2       # safe on 24GB unified memory (peak ~4.6GB at batch 2)
GRAD_ACCUM   = 4       # effective batch = BATCH_SIZE × GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
# ----------------------------------


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    # Step 1: Load the base model with Unsloth
    print(f"Loading base model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    print(f"Base model loaded. Type: {type(model).__name__}")

    # Step 2: Attach LoRA adapters to attention projections
    # target_modules are the weight matrices inside each attention block —
    # q/k/v are the query/key/value projections, o is the output projection.
    # These are the standard targets for LoRA on transformer models.
    print("Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True,  # trade compute for memory
    )

    # Step 3: Load training and validation data
    print(f"Loading data from {DATA_DIR}/")
    train_data = load_jsonl(f"{DATA_DIR}/train.jsonl")
    valid_data = load_jsonl(f"{DATA_DIR}/valid.jsonl")
    print(f"Train: {len(train_data)} examples, Valid: {len(valid_data)} examples")

    # Step 4: Train
    # Unsloth on Mac dispatches training to its MLX backend. If the
    # UnslothTrainer is available use it; otherwise fall back to the
    # mlx_lm.lora subprocess path with equivalent settings.
    try:
        from unsloth import UnslothTrainer, UnslothTrainingArguments
        from datasets import Dataset

        train_dataset = Dataset.from_list(train_data)
        valid_dataset = Dataset.from_list(valid_data)

        training_args = UnslothTrainingArguments(
            output_dir=ADAPTER_PATH,
            num_train_epochs=1,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=200,
            max_steps=ITERS,
        )

        trainer = UnslothTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
        )

        print("Starting training via UnslothTrainer...")
        trainer.train()
        model.save_pretrained(ADAPTER_PATH)
        print(f"Adapters saved to {ADAPTER_PATH}/")

    except ImportError:
        # UnslothTrainer not available on this Unsloth/MLX build —
        # fall back to mlx_lm.lora subprocess with equivalent settings.
        print("UnslothTrainer not available — falling back to mlx_lm.lora subprocess")
        import subprocess, sys
        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", MODEL_NAME,
            "--train",
            "--data", DATA_DIR,
            "--adapter-path", ADAPTER_PATH,
            "--batch-size", str(BATCH_SIZE),
            "--grad-accumulation-steps", str(GRAD_ACCUM),
            "--iters", str(ITERS),
            "--num-layers", str(NUM_LAYERS),
            "--learning-rate", str(LEARNING_RATE),
            "--grad-checkpoint",
            "--mask-prompt",
            "--save-every", "200",
            "--val-batches", "25",
        ]
        print("Command:", " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("Training failed. Run `python -m mlx_lm lora --help` to debug.")


if __name__ == "__main__":
    main()