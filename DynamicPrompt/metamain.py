# main.py
import streamlit as st
from metabot import MetaBot
from langchain_core.prompts import PromptTemplate

# -----------------------------------------------------
# Streamlit App: Research Paper Summarizer
# -----------------------------------------------------

st.set_page_config(page_title="Text Summarizer", layout="wide")
st.title("📘 Research Paper Summarizer")

# -----------------------------------------------------
# Initialize the model only once (avoid reloads)
# -----------------------------------------------------
if "bot" not in st.session_state:
    st.session_state.bot = MetaBot()

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

user_note = st.text_area("Enter your note (optional):", height=100)

# -----------------------------------------------------
# Build Prompt Template
# -----------------------------------------------------
template = PromptTemplate(
    template="""
Summarize the research paper titled "{paper_input}" with the following specifications:

Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain mathematical concepts with simple, intuitive code snippets.
2. Analogies:
   - Use relatable analogies to simplify complex ideas.
3. User Condition:
   - {user_note}

If certain information is not available in the paper, respond with: 
"Insufficient information available" instead of guessing.

Ensure the summary is clear, accurate, and aligned with the specified style and length.
""",
    input_variables=["paper_input", "style_input", "length_input", "user_note"]
)

prompt = template.format(
    paper_input=paper_input,
    style_input=style_input,
    length_input=length_input,
    user_note=user_note
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
