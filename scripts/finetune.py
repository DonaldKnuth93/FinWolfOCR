"""
Stage 4 — Fine-tune Qwen2.5-7B-Instruct with LoRA via PyTorch + PEFT
Intel Mac compatible (CPU training). Safe to run overnight.
"""
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" / "training"
ADAPTER_DIR = ROOT / "models" / "finwolf-7b-adapter"
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 7B fits comfortably in 64GB on CPU

# Intel CPU: use float32 for stability
DTYPE = torch.float32

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)


def check_data() -> int:
    train = DATA_DIR / "finwolf_train.jsonl"
    if not train.exists():
        print("❌  finwolf_train.jsonl not found. Run generate_training_data.py first.")
        sys.exit(1)
    count = sum(1 for _ in open(train))
    valid_count = 0
    if (DATA_DIR / "finwolf_valid.jsonl").exists():
        valid_count = sum(1 for _ in open(DATA_DIR / "finwolf_valid.jsonl"))
    print(f"✅  Training data: {count} train · {valid_count} valid examples")
    return count


def load_jsonl(path: Path) -> list:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                examples.append(json.loads(line))
            except Exception:
                pass
    return examples


def format_example(example: dict, tokenizer) -> dict:
    messages = example.get("messages", [])
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def tokenize(example: dict, tokenizer, max_length: int = 1024) -> dict:
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    count = check_data()
    iters = max(300, min(1000, count * 3))

    print(f"\n   Base model : {BASE_MODEL}")
    print(f"   Device     : CPU (Intel Mac)")
    print(f"   Dtype      : {DTYPE}")
    print(f"   Max steps  : {iters}")
    print(f"   Adapter    : {ADAPTER_DIR}")
    print("\nLoading tokenizer…")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (this may take a few minutes on first run — downloads ~15GB)…")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Load and format datasets
    print("\nPreparing datasets…")
    train_raw = load_jsonl(DATA_DIR / "finwolf_train.jsonl")
    train_formatted = [format_example(e, tokenizer) for e in train_raw]
    train_tokenized  = [tokenize(e, tokenizer) for e in train_formatted]
    train_dataset = Dataset.from_list(train_tokenized)

    eval_dataset = None
    valid_path = DATA_DIR / "finwolf_valid.jsonl"
    if valid_path.exists():
        valid_raw = load_jsonl(valid_path)
        valid_formatted = [format_example(e, tokenizer) for e in valid_raw]
        valid_tokenized  = [tokenize(e, tokenizer) for e in valid_formatted]
        eval_dataset = Dataset.from_list(valid_tokenized)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=str(ADAPTER_DIR),
        max_steps=iters,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # effective batch = 4
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        no_cuda=True,          # Intel CPU — no CUDA
        use_mps_device=False,  # no MPS on Intel
        fp16=False,            # CPU: use float32
        bf16=False,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\nStarting fine-tuning… (safe to leave running overnight)")
    print("Progress logs will appear every 10 steps.")
    print("-" * 60)

    trainer.train()

    print("\nSaving adapter weights…")
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    print(f"\n✅  Fine-tuning complete!")
    print(f"   Adapter saved to: {ADAPTER_DIR}")
    print("   Next step: python3 scripts/test_model.py")


if __name__ == "__main__":
    main()
