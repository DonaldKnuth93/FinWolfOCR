"""
Stage 7 — Export fine-tuned model as GGUF for VPS deployment
Merges LoRA adapter into base model, then converts to Q4_K_M GGUF.
Requires: pip3 install mlx-lm  +  llama.cpp installed via brew
"""
import subprocess
import sys
from pathlib import Path

ROOT         = Path(__file__).parent.parent
ADAPTER_DIR  = ROOT / "models" / "finwolf-14b-adapter"
MERGED_DIR   = ROOT / "models" / "finwolf-14b-merged"
GGUF_PATH    = ROOT / "models" / "finwolf-14b-q4.gguf"
BASE_MODEL   = "Qwen/Qwen2.5-14B-Instruct"


def check_deps():
    errors = []
    try:
        import mlx_lm  # noqa
    except ImportError:
        errors.append("mlx-lm not installed — run: pip3 install mlx-lm")

    result = subprocess.run(["which", "convert_hf_to_gguf.py"], capture_output=True)
    if result.returncode != 0:
        # Try the brew path
        llama_convert = Path("/usr/local/bin/convert_hf_to_gguf.py")
        if not llama_convert.exists():
            errors.append("llama.cpp not installed — run: brew install llama.cpp")

    if errors:
        for e in errors:
            print(f"❌  {e}")
        sys.exit(1)


def merge_adapter():
    if MERGED_DIR.exists() and list(MERGED_DIR.glob("*.safetensors")):
        print(f"✅  Merged model already exists at {MERGED_DIR}")
        return

    print("Merging LoRA adapter into base model…")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model",        BASE_MODEL,
        "--adapter-path", str(ADAPTER_DIR),
        "--save-path",    str(MERGED_DIR),
        "--de-quantize",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌  Merge failed")
        sys.exit(1)
    print(f"✅  Merged model saved to {MERGED_DIR}")


def convert_to_gguf():
    if GGUF_PATH.exists():
        print(f"✅  GGUF already exists at {GGUF_PATH}")
        return

    print("Converting to GGUF (Q4_K_M quantization)…")
    # Find convert script
    candidates = [
        "/usr/local/bin/convert_hf_to_gguf.py",
        "/opt/homebrew/bin/convert_hf_to_gguf.py",
    ]
    convert_script = next((p for p in candidates if Path(p).exists()), None)
    if not convert_script:
        result = subprocess.run(["which", "convert_hf_to_gguf.py"], capture_output=True, text=True)
        convert_script = result.stdout.strip() or None

    if not convert_script:
        print("❌  convert_hf_to_gguf.py not found. Install llama.cpp: brew install llama.cpp")
        sys.exit(1)

    cmd = [
        sys.executable, convert_script,
        str(MERGED_DIR),
        "--outfile", str(GGUF_PATH),
        "--outtype", "q4_k_m",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌  GGUF conversion failed")
        sys.exit(1)

    size_gb = GGUF_PATH.stat().st_size / 1e9
    print(f"✅  GGUF saved: {GGUF_PATH}  ({size_gb:.1f} GB)")


def print_deploy_instructions():
    print()
    print("=" * 60)
    print("VPS Deployment (Hostinger KVM4 / 16GB RAM)")
    print("=" * 60)
    print(f"1. Upload GGUF to VPS:")
    print(f"   scp {GGUF_PATH} user@your-vps:/opt/finwolf/")
    print()
    print("2. On the VPS, run llama.cpp server:")
    print("   llama-server -m /opt/finwolf/finwolf-14b-q4.gguf \\")
    print("     --host 0.0.0.0 --port 8080 --ctx-size 4096 -ngl 0")
    print()
    print("3. Update app/.env on VPS:")
    print("   LOCAL_MODEL_URL=http://localhost:8080")
    print()
    print("4. Restart Streamlit app — it will use the local model.")


def main():
    check_deps()

    if not ADAPTER_DIR.exists():
        print(f"❌  Adapter not found at {ADAPTER_DIR}. Run finetune.py first.")
        sys.exit(1)

    merge_adapter()
    convert_to_gguf()
    print_deploy_instructions()


if __name__ == "__main__":
    main()
