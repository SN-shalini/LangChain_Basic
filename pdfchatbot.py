import os
import base64
import torch
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI, Replicate
from langchain.callbacks import get_openai_callback
from sentence_transformers import SentenceTransformer

def display_pdf(uploaded_file):
            bytes_data = uploaded_file.getvalue()
            base64_pdf = base64.b64encode(bytes_data).decode("utf-8")
            pdf_display = (
            f'<embed src="data:application/pdf;base64,{base64_pdf}" width="500" height="800" type="application/pdf">'
            )
            return pdf_display

def clear_submit():
    st.session_state["submit"] = False

def extract_text_from_pdf(uploaded_file):
    # extract the text
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def data_chunking(text):
      # split into chunks
        chunks = []
        text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

def create_openai_embeddings(chunks):
      # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        return knowledge_base


def create_huggingface_embeddings(chunks):
      # create embeddings
        embeddings = HuggingFaceEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        return knowledge_base

if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["REPLICATE_API_TOKEN"] = ""
    st.set_page_config(page_title="PDF Chat Bot", layout="wide")
    st.header("PDF Chat Bot")


    with st.sidebar:
        model_name = st.radio("Select the model", ("OpenAI", "Llama2"))
        uploaded_file = st.file_uploader("Upload your PDF", type="pdf", on_change=clear_submit)

    col1, col2 = st.columns([6,5], gap = "small")

    if uploaded_file:
        with col1:
                pdf_text = extract_text_from_pdf(uploaded_file)
                st.write("Total words of the document: ", len(pdf_text.split(" ")))
                pdf_display = display_pdf(uploaded_file)
                st.markdown(pdf_display, unsafe_allow_html=True)
        

        with col2:
                pdf_text = extract_text_from_pdf(uploaded_file)
                chunks = data_chunking(pdf_text)

                if model_name == "OpenAI":
                    embeddings = create_openai_embeddings(chunks)
                    llm = OpenAI()
                else:
                    embeddings = create_huggingface_embeddings(chunks)
                    llm = Replicate(model="a16z-infra/llama-2-7b-chat:7b0bfc9aff140d5b75bacbed23e91fd3c34b01a1e958d32132de6e0a19796e2c",
                                    input={"temperature": 0.75, "max_length": 3000}
                                    )

                user_question = st.text_input("Ask a question about your paper:")
                if user_question:
                    docs = embeddings.similarity_search(user_question)

                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=user_question)
                    st.write(response)
