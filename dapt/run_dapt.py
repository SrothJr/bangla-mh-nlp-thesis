import os
import gc
import torch
import warnings
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint

# Ignore specific huggingface warnings for cleaner output
warnings.filterwarnings("ignore")


class CheckpointVerificationCallback(TrainerCallback):
    """
    After every checkpoint write, opens the safetensors header (fast — no full
    weights load) to verify the file is not corrupted.  If verification fails,
    a loud WARNING is printed while the PREVIOUS checkpoint is still safely on
    disk (save_total_limit >= 2), giving the user a clean fallback.
    """
    def on_save(self, args, state, control, **kwargs):
        import safetensors.torch as st
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        weights_file   = os.path.join(checkpoint_dir, "model.safetensors")
        if not os.path.exists(weights_file):
            return  # nothing to verify yet
        try:
            with st.safe_open(weights_file, framework="pt", device="cpu") as f:
                _ = list(f.keys())  # only reads the header, not the full tensors
            print(f"  ✔ Checkpoint verified OK: checkpoint-{state.global_step}")
        except Exception as e:
            print(
                f"\n  ⚠ WARNING: checkpoint-{state.global_step} appears CORRUPTED ({e}).\n"
                f"  The previous checkpoint is still intact — delete the bad one and resume.\n"
            )



# Directories
DATA_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/bangla_dapt_corpus_clean.txt"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/dapt_pretrained"))

MODEL_CONFIGS = {
    "BanglaBERT": {
        "model_id":   "csebuetnlp/banglabert",
        "train_batch": 128,   
        "eval_batch":  256,
        "grad_accum":  1,     
    },
    "sahajBERT": {
        "model_id":   "neuropark/sahajBERT",
        "train_batch": 32,    
        "eval_batch":  128,
        "grad_accum":  4,     
    },
    "mBERT": {
        "model_id":   "bert-base-multilingual-cased",
        "train_batch": 32,    # lowered to 32 because 64 spilled into Shared RAM
        "eval_batch":  32,
        "grad_accum":  4,     # effective batch = 32 × 4 = 128
    },
}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading corpus from {DATA_PATH}...")
    # Load the cleaned translation dataset
    dataset = load_dataset("text", data_files={"train": DATA_PATH})["train"]

    # Split dataset: 90% train, 10% validation (as specified in the thesis)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset   = split_dataset["test"]

    for model_name, model_cfg in MODEL_CONFIGS.items():
        model_id = model_cfg["model_id"]
        train_batch = model_cfg["train_batch"]
        eval_batch = model_cfg["eval_batch"]
        grad_accum = model_cfg["grad_accum"]

        print(f"\n{'='*60}\nStarting DAPT for: {model_name} ({model_id})\n{'='*60}")
        print(f"Batch config: train={train_batch}, eval={eval_batch}, grad_accum={grad_accum} -> effective batch={train_batch * grad_accum}")

        model_output_dir = os.path.join(OUTPUT_DIR, model_name)
        marker_file      = os.path.join(model_output_dir, "dapt_complete.txt")

        # Smart skipping — if DAPT already finished for this model, skip it
        if os.path.exists(marker_file):
            print(f"Skipping {model_name} — DAPT already completed.")
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # ------------------------------------------------------------------ #
        # FIX 1: max_length=256 to match the fine-tuning context window.
        # Previously 128 caused a context-length mismatch: the model adapted
        # its attention patterns to 128-token sequences during DAPT, then was
        # evaluated on 256-token sequences during fine-tuning — an unfair and
        # harmful inconsistency. Both stages must operate on the same window.
        # ------------------------------------------------------------------ #
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256
            )

        print(f"Tokenizing dataset for {model_name}...")
        # Single-process mapping (multiprocessing removed for Windows stability)
        tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_val   = val_dataset.map(tokenize_function,   batched=True, remove_columns=["text"])

        # Dynamic masking collator (15% masking probability — standard MLM)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        # Load the masked language model (ForMaskedLM, NOT SequenceClassification)
        model = AutoModelForMaskedLM.from_pretrained(model_id)

        # ------------------------------------------------------------------ #
        # FIX 2: Increased num_train_epochs from 10 → 30.
        # The previous run's MLM loss was still declining at epoch 9 with no
        # sign of convergence. EarlyStoppingCallback (patience=5) terminates
        # naturally when loss plateaus; 30 is a safe ceiling.
        #
        # FIX 3: Increased learning_rate from 2e-5 → 3e-5.
        # Slightly higher LR enables faster domain adaptation on the large
        # corpus. warmup_steps stabilises early training.
        #
        # FIX 4: Increased per_device_train_batch_size from 32 → 128.
        # The RTX 5090 (32 GB VRAM) was only using ~9.6 GB at batch=32.
        # Batch=128 fills ~28-30 GB — ~4× throughput with no performance cost.
        # For MLM pretraining (unsupervised), large batches are well-supported
        # by the literature (RoBERTa used batch=8,192). The "noisy gradients =
        # better regularization" argument applies to supervised fine-tuning,
        # not to unsupervised MLM on a 250K-document corpus.
        # warmup_steps scaled down proportionally: 225K / 128 = ~1,760
        # steps/epoch, so 500 steps ≈ same relative warmup as before.
        #
        # EarlyStoppingCallback patience raised from 3 → 5 to avoid stopping
        # too early on the longer training schedule.
        # ------------------------------------------------------------------ #
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            num_train_epochs=30,
            per_device_train_batch_size=train_batch,
            per_device_eval_batch_size=eval_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=3e-5,
            warmup_steps=500,   # Scaled from 2000: steps/epoch drops 7035→1760 at batch 128
            weight_decay=0.01,
            eval_strategy="steps",
            eval_steps=500,         # Eval every 500 steps (~2 min): fast, useful for loss monitoring and early stopping
            save_strategy="steps",
            save_steps=500,        # Save every 500 steps (~2 min): reduces disk I/O overhead from ~40% → ~10%
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            save_total_limit=3,    # Keep 3 checkpoints: ~50 min of fallback coverage at all times
            logging_steps=100,  # Increased logging frequency to match fewer steps/epoch
            seed=42,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=5),
                CheckpointVerificationCallback(),
            ]
        )

        # Resume from checkpoint if a previous run was interrupted, otherwise start fresh
        last_checkpoint = get_last_checkpoint(model_output_dir)
        if last_checkpoint is not None:
            print(f"Resuming {model_name} from checkpoint: {last_checkpoint}")
        else:
            print(f"Starting DAPT training for {model_name}...")
        trainer.train(resume_from_checkpoint=last_checkpoint)

        # Save the best DAPT model weights and tokenizer
        trainer.save_model(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)

        # Write completion marker so interrupted reruns skip this model cleanly
        with open(marker_file, "w") as f:
            f.write("DAPT completed successfully.\n")

        print(f"[{model_name}] DAPT complete! Weights saved to: {model_output_dir}")

        # ------------------------------------------------------------------ #
        # GPU CLEANUP: Explicitly free the model, optimizer states, and all
        # dataset tensors from GPU and CPU memory before the next model loads.
        # Without this, the previous model's weights + Adam states (~1.27 GB)
        # remain allocated on VRAM, causing OOM on the next iteration.
        # ------------------------------------------------------------------ #
        del trainer, model, tokenized_train, tokenized_val, data_collator, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        # Windows WDDM does not release GPU memory instantly after a process/model
        # is deleted — a brief pause lets the driver fully flush before the next
        # model loads, preventing false OOM errors on back-to-back model runs.
        import time
        time.sleep(10)
        print(f"GPU memory freed. Ready for next model.")

    print("\nAll Domain-Adaptive Pretraining finished successfully!")

if __name__ == "__main__":
    main()
