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

# Styling Configuration
st.set_page_config(page_title="BGC ChatBot", page_icon="🛢️", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #1E1E2E;
    color: #E0E0E0;
}
.stTextInput > div > div > input {
    background-color: #2C2C3E;
    color: #E0E0E0;
    border: 2px solid #4A6CF7;
    border-radius: 10px;
    padding: 10px;
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
    background-color: #1E1E2E;
    padding: 10px;
    box-shadow: 0 -2px 5px rgba(0,0,0,0.2);
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
    max-height: calc(100vh - 250px); /* Adjust based on your layout */
    overflow-y: auto;
    padding-bottom: 150px; /* Space for the fixed input section */
}
.chat-message {
    margin: 10px 0;
    padding: 10px;
    border-radius: 10px;
    background-color: #2C2C3E;
    max-width: 80%;
    word-wrap: break-word;
}
.chat-message.user {
    margin-left: auto;
    background-color: #4A6CF7;
}
.chat-message.assistant {
    margin-right: auto;
    background-color: #2C2C3E;
}
.supporting-info {
    margin-top: 20px;
    padding: 10px;
    background-color: #2C2C3E;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# API Configuration
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

def record_voice(language="en"):
    text = speech_to_text(
        start_prompt="🎤",
        stop_prompt="⏹️",
        language=language,
        use_container_width=True,
        just_once=True,
    )
    return text if text else None

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer questions based on the provided context about Basrah Gas Company but don't say in the answer about According to the provided text or pdf or bgc file just answer without tell us that."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("system", "Context: {context}"),
])

# Initialize Streamlit Sidebar
with st.sidebar:
    voice_language = st.selectbox("Voice Input Language", 
        ["English", "Arabic"])

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
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.markdown(f'<div class="chat-message {role}">{content}</div>', unsafe_allow_html=True)
    
    # Display supporting information for each assistant message
    if role == "assistant" and "supporting_info" in message:
        with st.expander("Supporting Information"):
            for i, doc in enumerate(message["supporting_info"]):
                page_number = doc.metadata.get("page", "unknown")
                st.write(f"**Document {i+1}** - Page: {page_number}")
                st.write(doc.page_content)
                st.write("--------------------------------")
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
    human_input = st.text_input("Ask something about the document", key="user_input", label_visibility="collapsed")

with col2:
    voice_input = record_voice(language=input_lang_code)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Process input
if voice_input:
    human_input = voice_input

if human_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": human_input})
    
    # Clear the input field
    st.session_state.user_input = ""

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({
            "input": human_input,
            "context": retriever.get_relevant_documents(human_input),
            "history": st.session_state.memory.chat_memory.messages
        })

        assistant_response = response["answer"]
        supporting_info = response["context"]

        # Add assistant response and supporting info to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
            "supporting_info": supporting_info
        })

        # Update memory
        st.session_state.memory.chat_memory.add_user_message(human_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)

        # Rerun to update the chat display
        st.rerun()
    else:
        assistant_response = "Error: Unable to load embeddings. Please check the embeddings folder."
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        st.rerun()
