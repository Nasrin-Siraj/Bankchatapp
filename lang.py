import streamlit as st
import openai
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# from langchain.chains.question_answering import load_qa_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pyngrok import ngrok
from streamlit_chat import message
import time
from dotenv import load_dotenv
import asyncio


# Load environment variables from .env file
load_dotenv()

# Set OpenAI key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

st.set_page_config(
    page_title="BankBot",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("📝BankBot")
with st.sidebar:
    # Display chat history in the sidebar
    if "history" in st.session_state and st.session_state.history:
        for i, entry in enumerate(st.session_state.history):
            role = "You" if entry["role"] == "user" else "Assistant"
            st.write(f"{i+1}. {role}: {entry['content']}")


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


st.sidebar.button("New Chat", on_click=clear_chat_history, type="primary")


if "messages" not in st.session_state.keys():
    # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome! I'm here to answer your Questions. What can I do to assist you?",
        }
    ]


@st.cache_resource(show_spinner=False)


# Function to load the documents and create chunks
def load_and_split_documents(pdf_folder_path):
    with st.spinner(
        text="Loading and indexing the  docs – hang tight! This should take 1-2 minutes."
    ):

        documents = []
        for file in os.listdir(pdf_folder_path):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder_path, file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        chunked_documents = text_splitter.split_documents(documents)
        return chunked_documents


# Function to initialize the Chroma vector store
def initialize_chroma_store(chunked_documents, embeddings_model, persist_directory):
    db = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings_model,
        persist_directory=persist_directory,
    )
    return db


# Load the embedding and LLM model
embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=200, temperature=0)

# Load and split the documents
pdf_folder_path = "data"
chunked_documents = load_and_split_documents(pdf_folder_path)

# Initialize the Chroma vector store
persist_directory = "test_index"
db = initialize_chroma_store(chunked_documents, embeddings_model, persist_directory)

# Load the database
vectordb = Chroma(
    persist_directory=persist_directory, embedding_function=embeddings_model
)

prompt_template = """
-You are a BankBot for ADIB, a helpful IslamicBanking assistant tasked to answer the user's questions. 
-If they greet you, You reply with Hello, How Can I help you?
-Use the following context to answer the question.Do not go outside of the context.
-If the input/Question is out of context say "I don't Know, This is not related to me"
Chat History: {chat_history}
Question: {input}
Context: {context}
Answer:
"""

prompt = PromptTemplate.from_template(template=prompt_template)

# Create conversational retrieval chain with custom prompt
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vectordb.as_retriever()
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Prompt for user input and save to chat history
if question := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": question})

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(
                f'<div style=" padding:15px; border-radius:10px; margin-bottom:10px;">{message["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style=" padding:10px; border-radius:10px; margin-bottom:10px;">{message["content"]}</div>',
                unsafe_allow_html=True,
            )


# Function to generate response using RAG chain
def predict(question):
    response = rag_chain.invoke(
        {"input": question, "chat_history": st.session_state.messages}
    )
    print(response)
    return response["answer"]


# If last message is not from assistant, generate a new response
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = predict(st.session_state.messages[-1]["content"])
            st.markdown(
                f'<div style="padding:10px; border-radius:10px; margin-bottom:10px;">{response}</div>',
                unsafe_allow_html=True,
            )
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)  # Add response to message history
            # Display feedback buttons for the new response
            feedback_col1, feedback_col2 = st.columns(2)
            with feedback_col1:
                if st.button("👍", key=f"like_{len(st.session_state.messages)-1}"):
                    st.session_state.feedback.append(
                        {
                            "index": len(st.session_state.messages) - 1,
                            "feedback": "like",
                        }
                    )
                    st.success("Thanks for your feedback!")
            with feedback_col2:
                if st.button("👎", key=f"dislike_{len(st.session_state.messages)-1}"):
                    st.session_state.feedback.append(
                        {
                            "index": len(st.session_state.messages) - 1,
                            "feedback": "dislike",
                        }
                    )
                    st.error("Thanks for your feedback!")
