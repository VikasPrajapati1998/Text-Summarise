# st_test.py
import streamlit as st

# Title
st.title("Streamlit Diagnostic — Basic UI")

# Input field for user's name
user_name = st.text_input("Enter your name:")

# Button to greet the user
if st.button("Say Hello"):
    if user_name.strip():
        st.success(f"Hello {user_name} 👋")
    else:
        st.warning("Please enter your name first!")

# Optional info text
st.write("If you see this and the input works, Streamlit rendering and interactivity are working correctly.")
