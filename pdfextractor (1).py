from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# Configure Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from multiple PDFs
def get_pdf_text(pdf_docs):
    """Extract text from all uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:  # Process each page in the PDF
            text += page.extract_text()
    return text


# Function to split text into smaller, manageable chunks
def get_text_chunks(text):
    """Split long text into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to save text chunks as vector embeddings
def get_vector_store(text_chunks):
    """Create a vector store for embedding-based search."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save the vector store locally


# Function to set up a conversational chain for Generative AI
def get_conversational_chain():
    """Create a conversational chain for answering questions."""
    prompt_template = """
    Answer the question as detailed as possible using the provided context. 
    If the answer is not in the context, reply with: 'answer is not available'. 
    Do not provide incorrect or made-up answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Function to handle user input and generate a response
def user_input(user_question):
    """Process the user's question and return an AI-generated response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)  # Find relevant chunks based on the query
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply:", response["output_text"])  # Display the AI's response in the app


# Main Streamlit application
def main():
    """Main function to define the Streamlit app."""
    st.set_page_config(page_title="Multiple PDF Analyzer")
    st.header("Chat with Multiple PDFs using Generative AI")

    # User enters a question
    user_question = st.text_input("Ask a question:")
    
    if user_question:  # If the user submits a question
        user_input(user_question)

    # Sidebar for file uploads
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files:", accept_multiple_files=True, type="pdf")
        
        if st.button("Submit and Proceed"):
            if pdf_docs:
                with st.spinner("Processing your files..."):
                    raw_text = get_pdf_text(pdf_docs)  # Extract text from PDFs
                    text_chunks = get_text_chunks(raw_text)  # Split text into chunks
                    get_vector_store(text_chunks)  # Save text chunks as vector embeddings
                st.success("Processing Completed!")
            else:
                st.error("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()
