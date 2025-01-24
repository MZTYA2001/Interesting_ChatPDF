import streamlit as st
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Placeholder for the BGC logo (replace with your actual image if needed)
bgc_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Example_logo.svg/640px-Example_logo.svg.png"

# Function to initialize the LLM
def init_llm():
    # Add initialization logic for your language model (if any)
    return True  # Placeholder for LLM initialization

def main():
    # Initialize LLM before using it
    llm = init_llm()
    if llm is None:
        st.stop()

    # Initialize Streamlit Sidebar
    with st.sidebar:
        st.title("Settings")
        voice_language = st.selectbox("Voice Input Language", ["English", "Arabic"])
        dark_mode = st.checkbox("Dark Mode", value=True)

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
    st.image(bgc_logo, width=200)
    st.title("Mohammed Al-Yaseen | BGC ChatBot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Clear chat history
    if st.button("Clear Chat History", key="clear-button"):
        st.session_state.messages = []
        st.session_state.memory.clear()

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "assistant" and content.startswith("Image:"):
            st.image("9.png", caption="9 Life-Saving Rules")
        else:
            st.markdown(f"<div class='chat-message {role}'>{content}</div>", unsafe_allow_html=True)

    # Process user input
    user_input = st.text_input("Ask something about the document or request an image", key="user_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        if "9 life-saving rules" in user_input.lower():
            # Provide text and image response for 9 Life-Saving Rules
            response = "The 9 Life-Saving Rules are key safety guidelines that must be followed to ensure workplace safety. Here is an image with the detailed rules:"
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(f"<div class='chat-message assistant'>{response}</div>", unsafe_allow_html=True)
            st.image("9.png", caption="9 Life-Saving Rules")
        else:
            response = f"Processing your input: {user_input}"  # Placeholder response
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(f"<div class='chat-message assistant'>{response}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
