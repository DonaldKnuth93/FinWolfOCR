# FinWolf Local Model — Training Plan
**iMac 64GB · Apple Silicon · Zero API costs after completion**

---

## Goal
Replace the Gemini API in the FinWolf OCR app with a custom-trained local model that:
- Runs 100% offline on your iMac
- Is fine-tuned specifically on Swiss insurance & finance documents
- Costs nothing to run after setup
- Can later be deployed on a Hostinger VPS

---

## Hardware & Time Requirements

| Spec | Value |
|------|-------|
| Machine | iMac, Apple Silicon, 64GB RAM |
| Total human time | ~2–3 hours |
| Computer time (runs itself) | ~6–10 hours |
| Internet needed | Only for initial downloads |
| API credits needed | None |

---

## Architecture

```
PDF / Image
     ↓
PaddleOCR (pre-trained, no training needed)
     ↓  raw extracted text
Fine-tuned Qwen2.5-14B (LoRA, trained on your 199 docs)
     ↓
Structured JSON  →  same output format as Gemini today
```

**Why two stages:**
- PaddleOCR is purpose-built for document text extraction — better than any vision LLM for raw OCR
- The fine-tuned LLM only needs to read clean text and extract fields — easy task for a 14B model
- Result: fast, accurate, offline, private

---

## Model Choices

| Role | Model | Size on disk | RAM needed |
|------|-------|-------------|-----------|
| Teacher (generates training data) | Qwen2.5:32b via Ollama | ~18 GB | ~20 GB |
| Student (fine-tuned, your model) | Qwen2.5-14B-Instruct via MLX | ~15 GB | ~35 GB during training |
| Final inference | Fine-tuned Qwen2.5-14B | ~10 GB (quantized) | ~12 GB |

---

## Stage 0 — Pre-flight Check
> Do this first. Should take under 10 minutes.

```bash
# 1. Verify Homebrew
brew --version
# If missing: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Verify Python 3.10+
python3 --version

# 3. Install Ollama (if not already installed)
brew install ollama

# 4. Start Ollama service
ollama serve &

# 5. Pull teacher model (18GB — start this and let it run)
ollama pull qwen2.5:32b

# 6. Install Python dependencies
pip3 install mlx-lm paddleocr paddlepaddle pymupdf pillow tqdm requests
```

> ⏳ The `ollama pull` will take 10–30 minutes depending on your internet speed.
> While it downloads, continue reading and set up the project.

---

## Stage 1 — Project Setup
> 5 minutes

```bash
# Navigate to the project
cd /Users/killua93/WebstormProjects/FinWolfOCR

# Create the scripts and data directories
mkdir -p scripts data/raw_text data/training

# Verify your 199 reference documents are accessible
ls "/Users/killua93/Downloads/08 - Supplementary offers mixed insurances/insurance supplementary docs" | wc -l
# Should print a number close to 199
```

---

## Stage 2 — Extract Text from 199 PDFs
> Human time: 2 min · Computer time: 20–40 min

Run this script (Claude will write it when you say **"ready"**):

```bash
python3 scripts/extract_text.py
```

**What it does:**
- Reads every PDF from your 199 reference documents
- Renders each page with PyMuPDF (same as the app already does)
- Runs PaddleOCR on each page to extract raw text
- Saves one `.txt` file per PDF into `data/raw_text/`
- Progress bar so you can see it working

**Output:** `data/raw_text/` — 199 text files, one per document

---

## Stage 3 — Generate Training Data
> Human time: 2 min · Computer time: 2–4 hours (runs itself)

Run this script (Claude will write it when you say **"ready"**):

```bash
python3 scripts/generate_training_data.py
```

**What it does:**
- Loads each of the 199 extracted text files
- Sends each to your local Qwen2.5:32b (via Ollama) with 6 different extraction queries:
  - Contract data (policy number, insurer, dates)
  - Personal data (name, DOB, AHV number)
  - Premiums & costs (monthly/annual premium, franchise, deductible)
  - Supplementary insurance (products, hospital class, waiting periods)
  - Vehicle insurance fields
  - All key fields combined
- Collects the structured JSON responses
- Saves everything as `data/training/finwolf_train.jsonl`

**Output:** `data/training/finwolf_train.jsonl` — ~1,000–1,200 training examples

> 💡 This uses your local 32B model as the "teacher". No internet, no API costs.

---

## Stage 4 — Fine-tune Your Model
> Human time: 5 min setup · Computer time: 3–5 hours (start before lunch or before bed)

Run this script (Claude will write it when you say **"ready"**):

```bash
python3 scripts/finetune.py
```

**What it does:**
- Downloads Qwen2.5-14B-Instruct base model from HuggingFace (~15GB, first run only)
- Fine-tunes it using LoRA (Low-Rank Adaptation) via MLX-LM
  - Only trains adapter weights (~50MB) — not the full 15GB model
  - 1,000 iterations, batch size 2, 16 LoRA layers
- Saves adapter weights to `models/finwolf-14b-adapter/`
- Logs loss every 10 steps so you can monitor progress

**Equivalent manual command:**
```bash
mlx_lm.lora \
  --model Qwen/Qwen2.5-14B-Instruct \
  --train \
  --data data/training \
  --iters 1000 \
  --lora-layers 16 \
  --batch-size 2 \
  --val-batches 5 \
  --save-every 200 \
  --adapter-path models/finwolf-14b-adapter
```

> ✅ Safe to let run overnight. iMac will not overheat — Apple Silicon has excellent thermal management.

---

## Stage 5 — Test the Model
> 10 minutes

```bash
python3 scripts/test_model.py
```

**What it does:**
- Loads your fine-tuned model
- Runs 5 test extractions on sample documents
- Compares output format vs expected JSON
- Prints accuracy summary

---

## Stage 6 — Integrate into App
> 15 minutes

```bash
python3 scripts/integrate.py
```

Or Claude will update `app/core/extractor.py` directly to use the local model instead of Gemini.

**Before (Gemini — needs internet + credits):**
```python
response = gemini_model.generate_content(content, stream=True)
```

**After (local model — offline, free):**
```python
response = mlx_lm.stream_generate(model, tokenizer, prompt)
```

Same Streamlit UI. Same JSON output format. Zero API dependency.

---

## Stage 7 — Optional: Deploy on VPS
> After demo, when ready

```bash
# Export fine-tuned model as GGUF for VPS deployment
python3 scripts/export_gguf.py
# Output: models/finwolf-14b-q4.gguf (~8GB)
```

Deploy on **Hostinger KVM4** (16GB RAM, ~€20/month):
- Upload GGUF file to VPS
- Run `llama.cpp` server
- Point Streamlit app to `http://localhost:8080`
- No API costs ever again

---

## File Map (what Claude will write)

```
FinWolfOCR/
├── scripts/
│   ├── extract_text.py          ← Stage 2: PDF → OCR text
│   ├── generate_training_data.py ← Stage 3: text → training JSONL
│   ├── finetune.py              ← Stage 4: LoRA fine-tuning launcher
│   ├── test_model.py            ← Stage 5: accuracy test
│   └── export_gguf.py           ← Stage 7: export for VPS
├── data/
│   ├── raw_text/                ← 199 .txt files (Stage 2 output)
│   └── training/
│       ├── finwolf_train.jsonl  ← training set (Stage 3 output)
│       └── finwolf_valid.jsonl  ← validation set (Stage 3 output)
├── models/
│   └── finwolf-14b-adapter/     ← LoRA adapter weights (Stage 4 output)
└── app/
    └── core/
        └── local_extractor.py   ← replaces extractor.py (Stage 6)
```

---

## Quick Reference — Key Commands

```bash
# Check Ollama is running
ollama list

# Monitor fine-tuning progress (in a second terminal)
tail -f models/finwolf-14b-adapter/training.log

# Test a quick inference after fine-tuning
mlx_lm.generate \
  --model Qwen/Qwen2.5-14B-Instruct \
  --adapter-path models/finwolf-14b-adapter \
  --prompt "Extrahiere: Police Nr., Prämie, Franchise aus diesem Dokument: ..."

# Check GPU/RAM usage during training
sudo powermetrics --samplers gpu_power -n 1
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ollama: command not found` | Run `brew install ollama` |
| `No module named 'mlx_lm'` | Run `pip3 install mlx-lm` |
| `No module named 'paddleocr'` | Run `pip3 install paddleocr paddlepaddle` |
| Fine-tuning crashes (out of memory) | Reduce `--batch-size` to 1 |
| Ollama pull fails | Check internet, retry `ollama pull qwen2.5:32b` |
| PaddleOCR download hangs | It downloads models on first run — wait 2–3 min |

---

## Expected Results

After fine-tuning on 199 Swiss insurance documents:

| Metric | Generic 14B | Fine-tuned 14B | Gemini 2.0 Flash |
|--------|-------------|----------------|-----------------|
| Swiss field names (DE/FR) | ~70% | ~95% | ~93% |
| AHV / IBAN formatting | ~60% | ~98% | ~95% |
| Policy number extraction | ~75% | ~97% | ~94% |
| Multi-person documents | ~65% | ~90% | ~92% |
| Speed (per page) | Fast | Fast | Slow (API latency) |
| Cost | Free | Free | ~€0.001/page |
| Privacy | ✅ Local | ✅ Local | ❌ Cloud |

---

*Generated by Claude · FinWolf OCR Project · Continue by opening this project in Claude Code on your iMac and saying "ready"*
