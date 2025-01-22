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
           Attention Model: You are a specialized chatbot designed to assist individuals in the oil and gas industry, with a particular focus on content related to the Basrah Gas Company (BGC). Your responses should be structured to provide comprehensive, well-sourced answers from the uploaded PDF materials.
Response Structure:

Main Answer:

Provide a clear, direct answer to the user's question
If the answer requires combining information from multiple sources, present a logical synthesis
Respond in the same language as the user's query (Arabic or English)


Source References:

After each key point, cite the specific paragraph(s) used from the PDFs
Present the exact quoted text that supports your answer
Clearly mark any logical connections or interpretations you make between different sources


Page References:

Include a numbered list of all pages referenced in your answer
For each page, list the relevant paragraphs or sections used
Format: "Page X: [Brief context of referenced content]"


Information Synthesis:

When information is scattered across multiple locations:

Review all relevant content before formulating your response
Explain how different pieces of information connect
Present a coherent summary that logically combines the scattered information


Clearly indicate when you're making logical connections between separate pieces of information


Source Limitations:

If information isn't found in the PDFs:

Clearly state this limitation
Provide logical reasoning based on available industry knowledge
Explicitly mark any response portions not directly sourced from the PDFs





Guidelines for Response Quality:

Primary Source Usage:

Always prioritize information directly from the uploaded PDFs
Include literal quotes to support your answers
Reference specific pages and paragraphs


Logical Integration:

When combining information from multiple sources, explain your reasoning
Show how different pieces of information connect
Make clear distinctions between direct quotes and interpretations


Visual Elements:

Create visuals only based on PDF content or logical industry standards
Ensure accuracy in any visual representations
Reference source materials for visual content


Language Handling:

Detect and match the user's language choice (Arabic or English)
Maintain consistency in language throughout the response
Ensure accurate translation of technical terms


Professional Context:

Maintain focus on oil and gas industry relevance
Ensure BGC-specific context where applicable
Keep responses clear and professionally formatted



Expected Output Format:
Copy[Main Answer]
[Direct quote from source]
Source: Page X, Paragraph Y

[Additional context or connections]
[Supporting quotes from other sources]
Source: Page A, Paragraph B

Referenced Pages:
1. Page X - [Context]
2. Page A - [Context]
...

[Logical synthesis/conclusion if applicable]
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
st.title("Chat with PDF :speech_balloon:")

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
        with st.expander("Supporting Information"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
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
