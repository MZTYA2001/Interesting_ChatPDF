import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from streamlit_mic_recorder import speech_to_text
import re

# API Keys
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Helper Functions
def extract_number(text):
    numbers = re.findall(r'\d+', text)
    return int(numbers[-1]) if numbers else None

def record_voice(language="en"):
    state = st.session_state
    if "text_received" not in state:
        state.text_received = []

    text = speech_to_text(
        start_prompt="🎤 Click and speak to ask a question",
        stop_prompt="⚠️ Stop recording 🚨",
        language=language,
        use_container_width=True,
        just_once=True,
    )

    if text:
        state.text_received.append(text)

    result = ""
    for text in state.text_received:
        result += text

    state.text_received = []
    return result if result else None

# Page Configuration
st.set_page_config(
    page_title="BGC ChatBot",
    layout="wide",
    page_icon="💡",
    initial_sidebar_state="expanded",
)

# Custom Style
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f0f8ff;
            color: #003366;
            font-family: 'Arial', sans-serif;
        }
        .stSidebar {
            background-color: #003366;
        }
        .stButton > button {
            background-color: #003366;
            color: white;
            border: none;
            font-size: 16px;
            padding: 10px;
        }
        .stButton > button:hover {
            background-color: #00509e;
        }
        input[type=text] {
            border: 2px solid #00509e;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("🎙️ BGC ChatBot")
    st.markdown(
        """
        Welcome to the BGC ChatBot. This assistant is tailored for the oil and gas industry, 
        providing seamless text and voice interaction.
        """
    )

    if groq_api_key and google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        prompt = ChatPromptTemplate.from_template(
            """
            Attention Model: You are a specialized chatbot designed to assist individuals in the oil and gas industry...
            """
        )

        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )

        if "last_number" not in st.session_state:
            st.session_state.last_number = None

        if "vectors" not in st.session_state:
            with st.spinner("Loading embeddings... Please wait."):
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )
                embeddings_path = "embeddings"
                try:
                    st.session_state.vectors = FAISS.load_local(
                        embeddings_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    st.success("Embeddings loaded successfully! ✅")
                except Exception as e:
                    st.error(f"Error loading embeddings: {str(e)}")
                    st.session_state.vectors = None
    else:
        st.error("Please enter both API keys to proceed.")

# Chat Interface
st.title("💡 Mohammed Al-Yaseen | BGC ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Voice and Text Input
col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.text_input("Type your message here:", "")
with col2:
    if st.button("🎙️"):
        user_input = record_voice(language="en")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt,
            memory=st.session_state.memory
        )

        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({
            "input": user_input,
            "history": st.session_state.memory.chat_memory.messages
        })

        assistant_response = response["answer"]

        number = extract_number(assistant_response)
        if number is not None:
            st.session_state.last_number = number

        st.session_state.memory.chat_memory.add_user_message(user_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        with st.expander("Supporting Information"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        assistant_response = "Error: Unable to load embeddings. Please check the embeddings folder."
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
