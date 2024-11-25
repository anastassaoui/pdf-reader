import streamlit as st
from streamlit_chat import message
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile

load_dotenv()

# Get Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

# Title
st.markdown("<h2 style='text-align: center;'>Chat with Your PDF: Powered by Llama3 & Groq API</h2>", unsafe_allow_html=True)

# Initialize LLM and Prompt
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

prompt = ChatPromptTemplate.from_template(
    """
    Use the following context to answer the question concisely:
    <context>
    {context}
    </context>
    Question: {input}
    Answer:
    """
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Initialize chat history
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # Initialize vector store
if "uploaded_pdf" not in st.session_state:
    st.session_state.uploaded_pdf = None  # Track the uploaded PDF file
if "pending_input" not in st.session_state:
    st.session_state.pending_input = ""  # Temporary storage for user input
if "processing" not in st.session_state:
    st.session_state.processing = False  # Prevent duplicate processing

def create_vector_db_out_of_the_uploaded_pdf_file(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_file.read())
        pdf_file_path = temp_file.name

    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name='BAAI/bge-small-en-v1.5', 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    loader = PyPDFLoader(pdf_file_path)
    text_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(text_documents)
    st.session_state.vector_store = FAISS.from_documents(document_chunks, st.session_state.embeddings)

def handle_user_input():
    user_input = st.session_state.pending_input.strip()
    if user_input:
        # Append user input to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Process the input
        if st.session_state.vector_store:
            retriever = st.session_state.vector_store.as_retriever()
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Invoke the chain with the user's input
            response = retrieval_chain.invoke({'input': user_input})
            bot_reply = response['answer']
        else:
            bot_reply = "Please upload a PDF and create the vector store before asking questions."

        # Append bot reply to chat history
        st.session_state.chat_history.append({"role": "bot", "content": bot_reply})

        # Clear pending input
        st.session_state.pending_input = ""

# Sidebar for PDF Upload and Embedding
st.sidebar.markdown("### Upload PDF and Create Vector DB")
pdf_input_from_user = st.sidebar.file_uploader("Upload the PDF file", type=['pdf'])

if pdf_input_from_user is not None:
    if st.sidebar.button("Create the Vector DB from the uploaded PDF file"):
        # Check if the file is new or has changed
        if (
            st.session_state.uploaded_pdf is None
            or st.session_state.uploaded_pdf.name != pdf_input_from_user.name
        ):
            st.session_state.uploaded_pdf = pdf_input_from_user  # Update the stored PDF
            create_vector_db_out_of_the_uploaded_pdf_file(pdf_input_from_user)
            st.sidebar.success("Vector Store DB for this PDF file is ready!")
        else:
            st.sidebar.info("Vector store already exists for the uploaded PDF.")
else:
    st.sidebar.warning("Please upload a PDF file to create a vector store.")

# Chat Interface
st.markdown("### Chat with Your PDF")

# Render chat history using streamlit-chat
for i, chat in enumerate(st.session_state.chat_history):
    role = chat["role"]
    content = chat["content"]
    is_user = role == "user"
    message(content, is_user=is_user, key=f"message_{i}")

# Input box with on_change trigger
st.text_input(
    "",
    value=st.session_state.pending_input,
    placeholder="Ask something about the PDF...",
    key="pending_input",
    on_change=handle_user_input
)
