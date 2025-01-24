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
from PIL import Image

# API Keys (Replace with your actual keys)
GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Load BGC Logo
bgc_logo = Image.open("BGC Logo.png")

# Styling Configuration
st.set_page_config(page_title="Mohammed Al-Yaseen | BGC ChatBot", page_icon="🤖", layout="wide")

# Custom CSS 
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0A0F24, #1A1F34);
    color: #FFFFFF;
}
.stTextInput > div > div > input {
    background-color: #1E1E2E;
    color: #FFFFFF;
    border: 2px solid #4A6CF7;
    border-radius: 12px;
    padding: 12px;
    width: 100%;
}
.mic-button {
    background-color: #4A6CF7;
    color: white;
    border: none;
    border-radius: 50%;
    width: 50px;
    height: 50px;
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
.sticky-input {
    position: fixed;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    background-color: #0A0F24;
    padding: 10px;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.5);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
}
.input-container {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
}
.chat-container {
    max-height: calc(100vh - 250px);
    overflow-y: auto;
    padding-bottom: 150px;
}
.chat-message {
    margin: 10px 0;
    padding: 12px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1E1E2E, #2C2C3E);
    max-width: 80%;
    word-wrap: break-word;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.chat-message.user {
    margin-left: auto;
    background: linear-gradient(135deg, #4A6CF7, #6382FF);
}
.chat-message.assistant {
    margin-right: auto;
    background: linear-gradient(135deg, #2C2C3E, #3C3C4E);
}
.supporting-info {
    margin-top: 20px;
    padding: 12px;
    background: linear-gradient(135deg, #1E1E2E, #2C2C3E);
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.clear-button {
    background-color: #FF4B4B;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 8px 16px;
    cursor: pointer;
    transition: background-color 0.3s;
}
.clear-button:hover {
    background-color: #FF6B6B;
}
.logo-container {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
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

    # Initialize Streamlit Sidebar
    with st.sidebar:
        st.title("Settings")
        voice_language = st.selectbox("Voice Input Language", ["English", "Arabic"])
        dark_mode = st.toggle("Dark Mode", value=True)

    # Initialize vectors
    if "vectors" not in st.session_state:
        with st.spinner("Loading embeddings... Please wait."):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                embeddings_path = "embeddings"
                st.session_state.vectors = FAISS.load_local(
                    embeddings_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                st.sidebar.write("Embeddings loaded successfully 🎉")
            except Exception as e:
                st.error(f"Error loading embeddings: {str(e)}")
                st.session_state.vectors = None

    # Initialize memory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )

    # Display BGC Logo
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image(bgc_logo, width=200)
    st.markdown('</div>', unsafe_allow_html=True)

    st.title("DeepSeek ChatBot 🤖")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Clear chat history
    if st.button("Clear Chat History", key="clear-button"):
        st.session_state.messages = []
        st.session_state.memory.clear()

    # Display chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        st.markdown(f'<div class="chat-message {role}">{content}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Process user input
    input_lang_code = "ar" if voice_language == "Arabic" else voice_language.lower()[:2]

    # Sticky input at the bottom
    st.markdown('<div class="sticky-input">', unsafe_allow_html=True)

    # Create a container for the input and voice button
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    # Text input and voice button in the same row
    col1, col2 = st.columns([0.85, 0.15])

    with col1:
        user_input = st.text_input("Ask something about the document", key="user_input", label_visibility="collapsed")

    with col2:
        voice_input = record_voice(language=input_lang_code)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Process input
    if voice_input:
        user_input = voice_input

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if "vectors" in st.session_state and st.session_state.vectors is not None:
            with st.spinner("Thinking..."):
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

if __name__ == "__main__":
    main()
