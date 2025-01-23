import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from streamlit_mic_recorder import speech_to_text

# Styling Configuration
st.set_page_config(page_title="BGC ChatBot", page_icon="🛢️", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #1E1E2E;
    color: #E0E0E0;
}
.mic-button {
    background-color: #4A6CF7;
    color: white;
    border: none;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: background-color 0.3s;
    margin-left: 10px;
}
.mic-button:hover {
    background-color: #6382FF;
}
#chat-input-container {
    position: fixed;
    bottom: 0;
    left: 20%;
    right: 20%;
    padding: 10px;
    background-color: #1E1E2E;
    border-top: 2px solid #4A6CF7;
    z-index: 999;
    display: flex;
    align-items: center;
}
#chat-input-container input {
    flex: 1;
    padding: 10px;
    margin-right: 10px;
    border-radius: 5px;
    border: 2px solid #4A6CF7;
    background-color: #2C2C3E;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# API Configuration
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """Answer questions based on the provided context about Basrah Gas Company.
    <context>{context}</context>
    Question: {input}
    """
)

# Initialize Sidebar
with st.sidebar:
    voice_language = st.selectbox("Voice Input Language", 
        ["Arabic", "English", "French", "Spanish"])

# Check API Keys and Initialize LLM
if groq_api_key and google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

    # Initialize memory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )

    # Initialize vectors
    if "vectors" not in st.session_state:
        with st.spinner("Loading embeddings... Please wait."):

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            embeddings_path = "embeddings"

            try:
                st.session_state.vectors = FAISS.load_local(
                    embeddings_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                st.sidebar.write("Embeddings loaded successfully 🎉")
            except Exception as e:
                st.error(f"Error loading embeddings: {str(e)}")
                st.session_state.vectors = None
else:
    st.error("Please enter both API keys to proceed.")

st.title("Mohammed Al-Yaseen | BGC ChatBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Determine language code for voice input
input_lang_code = "ar" if voice_language == "Arabic" else voice_language.lower()[:2]

# Voice Recording Function
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

# Voice input trigger
voice_input = record_voice(language=input_lang_code)

# Bottom input container

# Process input
human_input = voice_input or st.text_input("", key="user_input", label_visibility="collapsed")

if human_input:
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.markdown(human_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({
            "input": human_input,
            "history": st.session_state.memory.chat_memory.messages
        })

        assistant_response = response["answer"]

        st.session_state.memory.chat_memory.add_user_message(human_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Supporting Information
        with st.expander("Supporting Information"):
            if "context" in response:
                for i, doc in enumerate(response["context"]):
                    page_number = doc.metadata.get("page", "unknown")
                    st.write(f"**Document {i+1}** - Page: {page_number}")
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            else:
                st.write("No context available.")
    else:
        assistant_response = "Error: Unable to load embeddings. Please check the embeddings folder."
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
