# app.py
import streamlit as st
from chatbot import GemmaBot

# Title
st.title("Text Summarizer")

# Initialize the model only once (to avoid reloading on every rerun)
if "bot" not in st.session_state:
    st.session_state.bot = GemmaBot()

# User Query
user_query = st.text_area("Enter your text to summarize:", height=150)

# Button
if st.button("Submit"):
    if not user_query.strip():
        st.warning("⚠️ Please enter some text before summarizing.")
    else:
        with st.spinner("Generating summary..."):
            try:
                response = st.session_state.bot.run(user_query)
                st.success("✅ Summarization complete!")
                st.subheader("Summary:")
                st.write(response)
            except Exception as e:
                st.error(f"❌ Error during summarization: {e}")
