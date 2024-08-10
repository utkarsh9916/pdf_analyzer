
import os
import io
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()


genai.configure(api_key=os.getenv("Add your api key"))

def get_pdf_text(pdf_docs):
    text = ""
    if isinstance(pdf_docs, bytes):
        pdf_docs = [io.BytesIO(pdf_docs)]  # Convert single bytes input to a list

    for pdf in pdf_docs:
        if isinstance(pdf, bytes):  # If pdf is bytes, convert to BytesIO
            pdf = io.BytesIO(pdf)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
Answer the questions as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in the provided context, just say, "answer is not available in the context." Don't provide a wrong answer.
context:\n {context}?\n
Questions: \n {questions}\n

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def load_faiss_index(pickle_file):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index = FAISS.load_local(pickle_file, embeddings=embeddings, allow_dangerous_deserialization=True)
    return faiss_index

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = load_faiss_index("faiss_index")
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "questions": user_question})  # Use 'questions' instead of 'question'

    st.write("Answer:", response["output_text"])

def main():
    st.set_page_config(page_title="PDF Analyzer", page_icon="📝", layout="wide", initial_sidebar_state="expanded")

    # Sidebar configuration
    with st.sidebar:
        st.title("📝 PDF Analyzer Menu")
        st.markdown("---")
        st.markdown("Upload your PDF files and ask questions based on their content.")
        uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)  
        st.markdown("---")  

        if st.button("Process PDFs"):
            if uploaded_pdfs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(uploaded_pdfs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload at least one PDF file.")

    # Main content area
    
    st.title(" 📚Chat with Your PDFs")
    st.markdown("### Ask any question based on the content of the uploaded PDFs.")

    user_question = st.text_input("Enter your question here:")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Analyzing..."):
                answer = user_input(user_question)
                st.markdown("### Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a question.")

    # Footer
    st.markdown("---")
    st.markdown("Developed with 💻 by Utkarsh")

if __name__ == "__main__":
    main()
