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
import time

# API Keys (Replace with your actual keys)
GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Styling Configuration
st.set_page_config(page_title="BGC ChatBot", page_icon="🏭", layout="wide")

# Custom CSS for ChatGPT-like design
st.markdown("""
<style>
body {
    color: #333;
    background-color: #f4f4f4;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
}
.stApp {
    max-width: 800px;
    margin: 0 auto;
    background-color: white;
    padding: 20px;
    box-shadow: 0 0 15px rgba(0,0,0,0.1);
}
.chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #e5e5e5;
    padding-bottom: 10px;
    margin-bottom: 20px;
}
.chat-header h1 {
    margin: 0;
    font-size: 1.5rem;
    color: #333;
}
.chat-header .user-info {
    color: #666;
    font-size: 0.9rem;
}
.chat-container {
    height: 500px;
    overflow-y: auto;
    padding: 20px;
    background-color: #f9f9f9;
    border-radius: 8px;
    margin-bottom: 20px;
}
.chat-message {
    margin-bottom: 15px;
    display: flex;
    align-items: flex-start;
}
.chat-message .avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    margin-right: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
}
.chat-message .user .avatar {
    background-color: #10a37f;
    color: white;
}
.chat-message .assistant .avatar {
    background-color: #4a5568;
    color: white;
}
.chat-message .message-content {
    flex-grow: 1;
    background-color: white;
    padding: 10px 15px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.input-area {
    display: flex;
    align-items: center;
    background-color: white;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    padding: 10px;
}
.input-area input {
    flex-grow: 1;
    border: none;
    outline: none;
    margin-right: 10px;
}
.input-area button {
    background-color: #10a37f;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer questions based on the provided context about Basrah Gas Company without explicitly mentioning the source of information."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("system", "Context: {context}"),
])

def record_voice(language="en"):
    text = speech_to_text(
        start_prompt="🎤",
        stop_prompt="⏹️",
        language=language,
        use_container_width=True,
        just_once=True,
    )
    return text if text else None

def init_llm():
    """Initialize LLM with error handling"""
    if not GROQ_API_KEY or not GOOGLE_API_KEY:
        st.error("Missing API keys. Please set GROQ_API_KEY and GOOGLE_API_KEY.")
        return None

    try:
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="gemma2-9b-it")
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

def main():
    # Initialize LLM before using it
    llm = init_llm()
    if llm is None:
        st.stop()

    # Custom Header
    st.markdown("""
    <div class="chat-header">
        <h1>BGC Chatbot</h1>
        <div class="user-info">Mohammed Al-Yaseen | BGC</div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize vectors and memory
    if "vectors" not in st.session_state:
        with st.spinner("Loading document embeddings..."):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                embeddings_path = "embeddings"
                st.session_state.vectors = FAISS.load_local(
                    embeddings_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                st.error(f"Error loading embeddings: {str(e)}")
                st.session_state.vectors = None

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat display container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        st.markdown(f'''
        <div class="chat-message {role}">
            <div class="avatar">{role[0].upper()}</div>
            <div class="message-content">{content}</div>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Voice input setup
    voice_language = st.sidebar.selectbox("Voice Input Language", ["English", "Arabic"])
    input_lang_code = "ar" if voice_language == "Arabic" else voice_language.lower()[:2]

    # Input area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.8, 0.2])
    
    with col1:
        user_input = st.text_input("Ask something about BGC", key="user_input", label_visibility="collapsed")
    
    with col2:
        voice_input = record_voice(language=input_lang_code)

    st.markdown('</div>', unsafe_allow_html=True)

    # Process input
    if voice_input:
        user_input = voice_input

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        if "vectors" in st.session_state and st.session_state.vectors is not None:
            with st.spinner("Generating response..."):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                response = retrieval_chain.invoke({
                    "input": user_input,
                    "context": retriever.get_relevant_documents(user_input),
                    "history": st.session_state.memory.chat_memory.messages
                })

                assistant_response = response["answer"]

                st.session_state.memory.chat_memory.add_user_message(user_input)
                st.session_state.memory.chat_memory.add_ai_message(assistant_response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )

                # Rerun to refresh the chat display
                st.experimental_rerun()

        else:
            st.error("Unable to load document embeddings. Please check the system configuration.")

if __name__ == "__main__":
    main()
