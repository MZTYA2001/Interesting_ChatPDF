import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Sidebar configuration
with st.sidebar:
    # Validate API key inputs and initialize components if valid
    if groq_api_key and google_api_key:
        # Set Google API key as environment variable
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Initialize ChatGroq with the provided Groq API key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # Define the chat prompt template
        prompt = ChatPromptTemplate.from_template(
            """
         Role: You are an intelligent assistant designed to interact with a specific PDF document. Your role is to provide accurate, source-based answers to user questions. Follow these rules when crafting your responses:

Guidelines for Responses
Source Content Only

Base your answers exclusively on the content within the uploaded PDF. Avoid using external knowledge or making assumptions.
Combine Information Across Pages

If the answer spans multiple sections or pages, synthesize the relevant information into a single, cohesive response.
Always provide citations, referencing all the page numbers where the information was found.
Contextual Accuracy

Include direct excerpts from the PDF when appropriate.
If direct text is not available or insufficient, summarize the relevant information comprehensively.
Handling Incomplete or Missing Information

If the requested topic is not covered in the PDF, state this clearly and politely inform the user.
Citations with Numerical Clarity

Always cite the page numbers or section titles for every answer.
Use a structured format, such as "Page 5, Page 7-8", to list all pages contributing to the response.
Formatted and User-Friendly Responses

Use headings, bullet points, or paragraphs to organize your response for easy readability.
Ensure answers are concise yet comprehensive.
Language Flexibility

Respond in the user's preferred language (Arabic or English).
For bilingual queries, provide answers in both languages when necessary.
            <context>
            {context}
            </context>
            Question: {input}
            """
        )

        # Load existing embeddings from files
        if "vectors" not in st.session_state:
            with st.spinner("Loading embeddings... Please wait."):
                # Initialize embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )

                # Load existing FAISS index with safe deserialization
                embeddings_path = "embeddings"  # Path to your embeddings folder
                try:
                    st.session_state.vectors = FAISS.load_local(
                        embeddings_path,
                        embeddings,
                        allow_dangerous_deserialization=True  # Only use if you trust the source of the embeddings
                    )
                    st.sidebar.write("Embeddings loaded successfully :partying_face:")
                except Exception as e:
                    st.error(f"Error loading embeddings: {str(e)}")
                    st.session_state.vectors = None

    else:
        st.error("Please enter both API keys to proceed.")

# Main area for chat interface
st.title("Mohammed Al-Yaseen | BGC ChatBot")

# Initialize session state for chat messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user queries
if human_input := st.chat_input("Ask something about the document"):
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.markdown(human_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        # Create and configure the document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get response from the assistant
        response = retrieval_chain.invoke({"input": human_input})
        assistant_response = response["answer"]

        # Append and display assistant's response
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Display supporting information from documents
        # with st.expander("Supporting Information"):
        #     for i, doc in enumerate(response["context"]):
        #         st.write(doc.page_content)
        #         st.write("--------------------------------")
    else:
        # Error message if vectors aren't loaded
        assistant_response = (
            "Error: Unable to load embeddings. Please check the embeddings folder and ensure the files are correct."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
