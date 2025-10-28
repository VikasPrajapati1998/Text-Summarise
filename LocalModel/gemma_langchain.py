# ============================================================
# use_gemma_langchain_local.py
# ============================================================
"""
Runs the locally downloaded Gemma model with LangChain.
No API calls, no remote endpoints — pure local inference.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

# ------------------------------------------------------------
# Local Model Path
# ------------------------------------------------------------
LOCAL_MODEL_DIR = r"D:\Models\gemma-2b-it"  # same as previous script

if not os.path.exists(LOCAL_MODEL_DIR):
    raise FileNotFoundError(
        f"Local model not found in '{LOCAL_MODEL_DIR}'. "
        f"Please run 'download_gemma_model.py' first."
    )

print(f"🚀 Loading Gemma 2B model from: {LOCAL_MODEL_DIR}")

# ------------------------------------------------------------
# Load model and tokenizer locally
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# ------------------------------------------------------------
# Create a text-generation pipeline
# ------------------------------------------------------------
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

# ------------------------------------------------------------
# Wrap with LangChain
# ------------------------------------------------------------
llm = HuggingFacePipeline(pipeline=generator)

# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------
prompt = "Explain why Python is popular for AI development in 3 sentences."
print(f"\n💬 Prompt: {prompt}\n")

response = llm(prompt)

print("🧠 Model Response:")
print(response)
