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
import re

# Styling Configuration
st.set_page_config(page_title="BGC ChatBot", page_icon="🛢️", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #1E1E2E;
    color: #E0E0E0;
}
/* Fixed input area */
.fixed-input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #252535;
    padding: 10px;
    z-index: 1000;
}
.input-row {
    display: flex;
    align-items: center;
    gap: 10px;
}
.voice-button {
    background-color: transparent;
    border: none;
    color: #4A6CF7;
    font-size: 24px;
    cursor: pointer;
    transition: color 0.3s;
}
.voice-button:hover, .voice-button:active {
    color: #6382FF;
}
.recording {
    color: red !important;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
.stChatInputContainer {
    margin-bottom: 80px;
}
</style>
""", unsafe_allow_html=True)

# API Configuration
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

def extract_number(text):
    numbers = re.findall(r'\d+', text)
    return int(numbers[-1]) if numbers else None

# Voice Recording Function
def record_voice(language):
    text = speech_to_text(
        start_prompt="🎤", 
        stop_prompt="🎤", 
        language=language,
        use_container_width=True,
        just_once=True,
    )
    return text

# Initialize Streamlit
st.title("Mohammed Al-Yaseen | BGC ChatBot")

# Sidebar Language Selector
with st.sidebar:
    voice_language = st.selectbox("Voice Input Language", 
        ["Arabic", "English", "French", "Spanish"])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Prepare LLM and Retrieval Setup
if groq_api_key and google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
    
    # Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """Answer questions based on the provided context about Basrah Gas Company.
        <context>{context}</context>
        Question: {input}
        """
    )

    # Initialize memory and vectors
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )

    if "vectors" not in st.session_state:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings_path = "embeddings"
        try:
            st.session_state.vectors = FAISS.load_local(
                embeddings_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            st.session_state.vectors = None

# Input Area with Voice Option
def process_input(human_input):
    if "vectors" in st.session_state and st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({
            "input": human_input,
            "history": st.session_state.memory.chat_memory.messages
        })
        
        assistant_response = response["answer"]
        
        st.session_state.messages.append({"role": "user", "content": human_input})
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        st.session_state.memory.chat_memory.add_user_message(human_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)
        
        st.experimental_rerun()
    else:
        st.error("Unable to load embeddings")

# Main input handling
input_lang_code = "ar" if voice_language == "Arabic" else voice_language.lower()[:2]
human_input = st.chat_input("Ask something about the document")
voice_input = record_voice(input_lang_code) if st.button("🎤", key="voice_button") else None

# Process input
if human_input:
    process_input(human_input)
elif voice_input:
    process_input(voice_input)
