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

# Styling Configuration
st.set_page_config(page_title="DeepSeek ChatBot", page_icon="🤖", layout="wide")

# Custom CSS (previous CSS remains the same)
st.markdown("""
<style>
... (previous CSS remains unchanged)
</style>
""", unsafe_allow_html=True)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer questions based on the provided context about Basrah Gas Company without explicitly mentioning the source of information."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("system", "Context: {context}"),
])

def init_llm():
    """Initialize LLM with error handling"""
    # Use environment variables or Streamlit secrets
    groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")

    if not groq_api_key or not google_api_key:
        st.error("Missing API keys. Please set GROQ_API_KEY and GOOGLE_API_KEY.")
        return None

    try:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        return ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

def record_voice(language="en"):
    text = speech_to_text(
        start_prompt="🎤",
        stop_prompt="⏹️",
        language=language,
        use_container_width=True,
        just_once=True,
    )
    return text if text else None

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
