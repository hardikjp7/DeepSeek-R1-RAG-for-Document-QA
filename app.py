import os
import streamlit as st
from rag_utility import process_document_to_chroma_db, answer_question

# Set the working directory
working_dir = os.getcwd()

st.title("🐋 DeepSeek-R1 - Document RAG")

# Add small text below the header
st.markdown("Made by 😎 [Hardik](https://www.linkedin.com/in/hardikjp/)")

# Sidebar for API key input
st.sidebar.header("API Configuration")
api_key = st.sidebar.text_input("Enter your API key", type="password")
# Guide for obtaining Google API Key if not available
st.sidebar.subheader("Don't have a Groq API Key?")
st.sidebar.write("Visit [here](https://console.groq.com/) and get your free API key")

if api_key:
    # File uploader widget
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Define save path
        save_path = os.path.join(working_dir, uploaded_file.name)
        # Save the file
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        process_document_to_chroma_db(uploaded_file.name, api_key)
        st.info("Document Processed Successfully")

    # Text widget to get user input
    user_question = st.text_area("Ask your question about the document")

    if st.button("Answer"):
        answer = answer_question(user_question, api_key)
        st.markdown("### DeepSeek-R1 Response")
        st.markdown(answer)
else:
    st.warning("Please enter your API key in the sidebar.")
