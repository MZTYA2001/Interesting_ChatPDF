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

# [Rest of the existing imports and initial setup remains the same]

# Add this CSS to fix the input container at the bottom
st.markdown("""
    <style>
    .fixed-bottom-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background-color: white;
        padding: 10px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stApp {
        padding-bottom: 80px;  /* Add padding to bottom to prevent content being hidden */
    }
    </style>
""", unsafe_allow_html=True)

# Modified voice recording function
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

# Main app logic
def main():
    # [Previous setup code remains the same]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Determine language code for voice input
    input_lang_code = "ar" if voice_language == "Arabic" else voice_language.lower()[:2]

    # Create a fixed bottom container for input
    with st.container():
        st.markdown('<div class="fixed-bottom-container">', unsafe_allow_html=True)
        
        # Use columns to center the input
        col1, col2, col3 = st.columns([1, 6, 1])
        
        with col2:
            # Voice input trigger
            voice_input = record_voice(language=input_lang_code)

            # Bottom input container
            human_input = voice_input or st.text_input("", key="user_input", label_visibility="collapsed")

        st.markdown('</div>', unsafe_allow_html=True)

    # Process input
    if human_input:
        st.session_state.messages.append({"role": "user", "content": human_input})
        with st.chat_message("user"):
            st.markdown(human_input)

        if "vectors" in st.session_state and st.session_state.vectors is not None:
            # [Rest of the processing logic remains the same]

# Run the main function
if __name__ == "__main__":
    main()
