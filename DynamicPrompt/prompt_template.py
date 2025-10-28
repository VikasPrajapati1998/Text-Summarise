# main.py
import streamlit as st
from gemmabot import GemmaBot
from langchain_core.prompts import load_prompt

# -----------------------------------------------------
# Streamlit App: Research Paper Summarizer
# -----------------------------------------------------

st.set_page_config(page_title="Text Summarizer", layout="wide")
st.title("📘 Research Paper Summarizer")

# -----------------------------------------------------
# Initialize the model only once (avoid reloads)
# -----------------------------------------------------
if "bot" not in st.session_state:
    st.session_state.bot = GemmaBot()

# -----------------------------------------------------
# Input Fields
# -----------------------------------------------------
paper_input = st.selectbox(
    "Select Research Paper Name",
    ["Select...", 
     "Attention is all you need", 
     "BERT: Pre-training of Deep Bidirectional Transformers", 
     "GPT-3: Language Models are Few-Short Learners", 
     "Diffusion Models Beat GANs on Image Synthesis."]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Select...", "Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Select...", "Short (1-2 Paragraphs)", "Medium (3-5 Paragraphs)", "Long (Detailed Explanation)"]
)

# -----------------------------------------------------
# Build Prompt Template
# -----------------------------------------------------
template = load_prompt('template.json')
prompt = template.format(
    paper_input=paper_input,
    style_input=style_input,
    length_input=length_input
)

# -----------------------------------------------------
# Generate Summary
# -----------------------------------------------------
if st.button("Submit"):
    if (
        paper_input == "Select..."
        or style_input == "Select..."
        or length_input == "Select..."
    ):
        st.warning("⚠️ Please select all the given options before submitting.")
    else:
        with st.spinner("🔄 Generating summary..."):
            try:
                response = st.session_state.bot.run(prompt)
                st.success("✅ Summarization complete!")
                st.subheader("📝 Summary Output:")
                st.write(response)
            except Exception as e:
                st.error(f"❌ Error during summarization: {e}")
