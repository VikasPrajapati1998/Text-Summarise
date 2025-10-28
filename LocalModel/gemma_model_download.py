# gemma_model_download.py
"""
Downloads and caches the Gemma 2 2B IT model from Hugging Face.
- Loads HUGGINGFACE_HUB_TOKEN from environment or .env
- Verifies access to gated repo before attempting large downloads
- Saves tokenizer and model to LOCAL_DIR
"""

import os
import sys
from pathlib import Path

# dotenv to load .env (optional)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

MODEL_ID = "google/gemma-2-2b-it"
LOCAL_DIR = Path(r"D:\Study\AI\Text-Summarise\LocalModel\Models\gemma-2b-it")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# Read token from env
HF_TOKEN = (
    os.getenv("HUGGINGFACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_TOKEN")
    or os.getenv("HF_TOKEN")
)

print("=== Gemma model downloader ===")
print(f"Target model: {MODEL_ID}")
print(f"Local save directory: {LOCAL_DIR}")
print("")

if not HF_TOKEN:
    print("⚠️  No Hugging Face token found in environment (.env or env vars).")
    print("    Please create a token at: https://huggingface.co/settings/tokens (with 'read' scope)")
    print("    and either:")
    print("      - run `huggingface-cli login` in this environment, OR")
    print("      - set HUGGINGFACE_HUB_TOKEN in your .env or system environment, then re-run this script.")
    sys.exit(1)

print("🔒 Using Hugging Face token from environment (will not be printed).")

# Import optional libs and handle missing packages gracefully
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import hf_hub_download
except Exception as e:
    print("❌ Required libraries not installed or import failed.")
    print("   Install with: pip install transformers huggingface_hub torch python-dotenv")
    raise

# Quick access check to the gated repo by attempting to fetch config.json metadata
print("\n🔎 Checking access to gated repo (this is fast) ...")
try:
    # this will raise if repo is gated or token lacks permission
    hf_hub_download(repo_id=MODEL_ID, filename="config.json", token=HF_TOKEN, repo_type="model")
    print("✅ Access check passed. Token can read model metadata.")
except Exception as e:
    print("⛔ Access check failed.")
    print("   Error:", type(e).__name__, str(e))
    sys.exit(1)

print("\n⬇️  Beginning download of tokenizer and model (this may take time & bandwidth)...")

# Decide device mapping and dtype
device_map = "auto"
load_kwargs = {"device_map": device_map}

# If CUDA is available prefer bfloat16 when possible (best-effort)
if torch.cuda.is_available():
    try:
        # bfloat16 may not be supported on all GPUs; transformers will handle if unsupported
        load_kwargs["torch_dtype"] = torch.bfloat16
        print("GPU detected -> requesting bfloat16 dtype (if supported).")
    except Exception:
        print("GPU detected but couldn't set bfloat16; using default dtype.")
else:
    # CPU-only: do not set torch_dtype to bfloat16; leave default to avoid incompatibility
    print("No GPU detected -> loading for CPU (this will be slower).")

# Perform downloads using use_auth_token so huggingface_hub knows to use the token
try:
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
    tokenizer.save_pretrained(LOCAL_DIR)
    print(f"Tokenizer saved to: {LOCAL_DIR}")

    print("Downloading model (this can be large)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN, **load_kwargs)
    # Save model weights/config locally
    model.save_pretrained(LOCAL_DIR)
    print(f"Model saved to: {LOCAL_DIR}")

    print("\n✅ Download complete. You can now run offline inference pointing to this folder.")
    print("Example path to use in your scripts:", str(LOCAL_DIR))
except Exception as exc:
    # print helpful error and then the original exception for debugging
    import traceback
    print("\n❌ Failed while downloading model or tokenizer.")
    traceback.print_exc()
    sys.exit(1)
