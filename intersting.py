import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

with st.sidebar:
    if groq_api_key and google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            ("system", """Attention Model: You are a specialized chatbot designed to assist individuals in the oil and gas industry, with a particular focus on content related to the Basrah Gas Company (BGC). Your responses must primarily rely on the PDF files uploaded by the user, which contain information specific to the oil and gas sector and BGC's operational procedures.
When providing responses, you must:

Structure Your Answer:
Begin with a clear, direct answer to the question
Support each point with relevant quotes from the PDFs
After each quote, cite the specific page number in format: [Page X]
List all referenced pages at the end of your response

Source Attribution Format:
Answer: [Your main response point]

Supporting Evidence:
"[Exact quote from PDF]" [Page X]

Additional Context:
"[Related quote from another section]" [Page Y]

Referenced Pages:
- Page X: [Brief description of content]
- Page Y: [Brief description of content]

Remember to maintain context from our previous conversation while providing accurate, sourced responses.

Context: {context}
Question: {input}"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )

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

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if human_input := st.chat_input("Ask something about the document"):
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

        chat_history = [
            (msg["content"], st.session_state.messages[i+1]["content"])
            for i, msg in enumerate(st.session_state.messages[:-1:2])
        ]
        
        st.session_state.memory.chat_memory.add_user_message(human_input)
        
        response = retrieval_chain.invoke({
            "input": human_input,
            "history": st.session_state.memory.chat_memory.messages
        })
        
        assistant_response = response["answer"]
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
