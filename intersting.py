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

# Custom Styling
st.set_page_config(
    page_title="Basrah Gas Company ChatBot", 
    page_icon="🛢️", 
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #E6F2FF;
}
.stTextInput > div > div > input {
    background-color: white;
    border: 2px solid #1E90FF;
    border-radius: 10px;
}
.mic-button {
    background-color: #1E90FF;
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
}
.mic-button:hover {
    background-color: #4169E1;
}
</style>
""", unsafe_allow_html=True)

# API Keys
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

def extract_number(text):
    numbers = re.findall(r'\d+', text)
    return int(numbers[-1]) if numbers else None

def record_voice(language="en"):
    state = st.session_state

    if "text_received" not in state:
        state.text_received = []

    text = speech_to_text(
        start_prompt="🎤 Click and speak to ask question",
        stop_prompt="⚠️Stop recording🚨",
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

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
Attention Model: You are a specialized chatbot designed to assist individuals in the oil and gas industry, with a particular focus on content related to the Basrah Gas Company (BGC). 
Your responses must primarily rely on the PDF files uploaded by the user, which contain information specific to the oil and gas sector and BGC's operational procedures. 
If a specific answer cannot be directly found in the PDFs, you are permitted to provide a logical and well-reasoned response based on your internal knowledge. 
Under no circumstances should you use or rely on information from external sources, including the internet.

Guidelines:
1. **Primary Source Referencing:**
- Always reference the specific page number(s) in the uploaded PDFs where relevant information is found. 
If the PDFs contain partial or related information, integrate it with logical reasoning to provide a comprehensive response. 
Clearly distinguish between PDF-derived content and logical extrapolations to ensure transparency.
Additionally, explicitly mention the total number of pages referenced in your response.

2. **Logical Reasoning:**
- When specific answers are unavailable in the PDFs, use your internal knowledge to provide logical, industry-relevant responses. 
Explicitly state when your response is based on reasoning rather than the uploaded materials.

3. **Visual Representation:**
- When users request visual representations (e.g., diagrams, charts, or illustrations), create accurate and relevant visuals based on the uploaded PDF content and logical reasoning. 
Ensure the visuals align precisely with the context provided and are helpful for understanding the topic. 
Provide an appropriate photo or diagram in the response if needed to enhance understanding, even if the user does not explicitly request it.

4. **Restricted Data Usage:**
- Avoid using or assuming information from external sources, including the internet or any pre-existing external knowledge that falls outside the uploaded materials or your internal logical reasoning.

5. **Professional and Contextual Responses:**
- Ensure responses remain professional, accurate, and relevant to the oil and gas industry, with particular tailoring for Basrah Gas Company. 
Maintain a helpful, respectful, and clear tone throughout your interactions.

6. **Multilingual Support:**
- Detect the language of the user's input (Arabic or English) and respond in the same language. 
If the input is in Arabic, provide the response in Arabic. If the input is in English, provide the response in English.

Expected Output:
- Precise and accurate answers derived from the uploaded PDFs, with references to specific page numbers where applicable. 
Include the total number of pages referenced in your response.
- Logical and well-reasoned responses when direct answers are not available in the PDFs, with clear attribution to reasoning.
- Accurate visual representations (when requested) based on PDF content or logical reasoning. Provide a relevant photo or diagram if it enhances understanding.
- Polite acknowledgments when information is unavailable in the provided material, coupled with logical insights where possible.
- Responses in the same language as the user's input (Arabic or English).

Thank you for your accuracy, professionalism, and commitment to providing exceptional assistance tailored to the Basrah Gas Company and the oil and gas industry.

Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
)

# Initialize Streamlit Sidebar
with st.sidebar:
    if groq_api_key and google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

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
                    st.sidebar.write("Embeddings loaded successfully :partying_face:")
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

# Voice input section
col1, col2 = st.columns([0.9, 0.1])

with col1:
    # Text input field
    human_input = st.text_input("Ask something about the document", key="user_input")

with col2:
    # Microphone button
    st.markdown("""
    <div class="mic-button" onclick="document.getElementById('voice_trigger').click()">🎤</div>
    <input type="hidden" id="voice_trigger">
    """, unsafe_allow_html=True)
    
    # Voice input trigger
    voice_input = record_voice(language="ar")  # Default to Arabic

# Determine input source
if voice_input:
    human_input = voice_input

# Process input
if human_input:
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.markdown(human_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(
            llm,
            prompt,
            memory=st.session_state.memory
        )
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({
            "input": human_input,
            "history": st.session_state.memory.chat_memory.messages
        })
        
        assistant_response = response["answer"]
        
        # Update last number if response contains a numerical result
        number = extract_number(assistant_response)
        if number is not None:
            st.session_state.last_number = number
            
        st.session_state.memory.chat_memory.add_user_message(human_input)
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
