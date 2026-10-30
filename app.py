import streamlit as st
import tempfile
import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


st.set_page_config(page_title="ContextAI", page_icon="📑", layout="wide")

st.title("📑 ContextAI")
st.caption("Your PDFs. Your Questions. Clear Answers.")

# ---------------------------------
# SESSION STATE
# ---------------------------------

defaults = {
    "authenticated": False,
    "api_key": "",
    "model": "",
    "pdf_processed": False,
    "vectorstore": None,
    "chat_history": [],
    "pdf_name": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------
# EMBEDDING MODEL
# ---------------------------------


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------
# GROQ AUTHENTICATION
# ---------------------------------

if not st.session_state.authenticated:

    st.subheader("🔐 Groq API Authentication")

    st.write("Enter your Groq API key and select the model to use ContextAI.")

    with st.form("authentication_form"):

        api_key = st.text_input(
            "Groq API Key", type="password", placeholder="Enter your Groq API key"
        )

        model = st.text_input(
            "Enter Groq Model Name", placeholder="Enter your Mode name"
        )

        authenticate = st.form_submit_button("Authenticate", use_container_width=True)

    if authenticate:

        if not api_key.strip():
            st.error("Please enter your API key.")

        else:
            try:
                with st.spinner("Authenticating..."):

                    test_llm = ChatGroq(
                        model=model, api_key=api_key, temperature=0.3, max_tokens=100
                    )

                    test_llm.invoke("Reply with OK.")

                st.session_state.api_key = api_key
                st.session_state.model = model
                st.session_state.authenticated = True

                st.success("Authentication successful!")

                st.rerun()

            except Exception as e:
                st.error(f"Authentication failed: {e}")


# ---------------------------------
# PDF UPLOAD SECTION
# ---------------------------------

elif not st.session_state.pdf_processed:

    st.subheader("📄 Upload Your PDF")

    st.write("Upload a PDF document to start chatting with ContextAI.")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:

        st.info(f"Selected file: {uploaded_file.name}")

        if st.button("Process PDF", use_container_width=True):

            try:
                with st.spinner("Reading PDF and creating your knowledge base..."):

                    # Save uploaded PDF temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    try:
                        # Load PDF
                        loader = PyPDFLoader(temp_path)
                        docs = loader.load()

                        # Split PDF into chunks
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, chunk_overlap=200
                        )

                        chunks = splitter.split_documents(docs)

                        if not chunks:
                            raise ValueError("No readable text was found in this PDF.")

                        # Create embeddings
                        embedding_model = get_embedding_model()

                        # Create in-memory Chroma database
                        vectorstore = Chroma.from_documents(
                            documents=chunks,
                            embedding=embedding_model,
                            collection_name=f"contextai_{uuid4().hex}",
                        )

                        st.session_state.vectorstore = vectorstore
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.chat_history = []

                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                st.success("PDF processed successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"PDF processing failed: {e}")


# ---------------------------------
# CHATBOT SECTION
# ---------------------------------

else:

    st.subheader("💬 Chat with Your PDF")

    st.success(f"Currently chatting with: {st.session_state.pdf_name}")

    st.caption(f"Model: {st.session_state.model}")

    # Initialize LLM
    llm = ChatGroq(
        model=st.session_state.model,
        api_key=st.session_state.api_key,
        temperature=0.3,
        max_tokens=500,
    )

    # Initialize retriever
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            You are ContextAI, a friendly, funny, and intelligent
            PDF AI assistant. Always introduce yourself to buld a friendly relationship.

            Rules:
            1. Answer using ONLY the provided PDF context.
            2. Never invent facts or unsupported claims.
            3. Use simple English.
            4. Keep answers concise unless more detail is requested.
            5. Explain difficult concepts with simple examples.
            6. Use light humor and emojis when appropriate.
            7. Include PDF page references for factual answers.
            8. Never invent page numbers.
            9. If the answer is not in the PDF, say:

            I couldn't find that information in your PDF.
            I suggest you not to get distracted from the topic! 😄

            10. Respond naturally to greetings and thanks.
            11. For unrelated factual questions, explain that
                you are a PDF-based assistant.
            12. Use conversation history to understand follow-up
                questions, but answer factual questions only
                from the provided PDF context.
            """,
            ),
            (
                "human",
                """
            Chat history:
            {history}

            PDF context:
            {context}

            Question:
            {question}

            Provide your answer with PDF page references.
            """,
            ),
        ]
    )

    # Display existing chat history
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    query = st.chat_input("Ask something about your PDF...")

    if query:

        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        try:
            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    # Retrieve relevant documents
                    docs = retriever.invoke(query)

                    # Add page metadata to context
                    context_parts = []

                    for doc in docs:

                        page = doc.metadata.get("page")

                        page_number = page + 1 if page is not None else "Unavailable"

                        context_parts.append(f"""
                            Source: {st.session_state.pdf_name}
                            Page: {page_number}

                            Content:
                            {doc.page_content}
                            """)

                    context = "\n\n".join(context_parts)

                    # Format previous conversation
                    history = ""

                    for message in st.session_state.chat_history[-7:-1]:

                        role = "User" if message["role"] == "user" else "ContextAI"

                        history += f"{role}: {message['content']}\n"

                    # Generate response
                    final_prompt = prompt.invoke(
                        {"context": context, "question": query, "history": history}
                    )

                    response = llm.invoke(final_prompt)

                    answer = response.content

                    st.markdown(answer)

                    # Save assistant response
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

        except Exception as e:
            st.error(f"Something went wrong: {e}")

            # Remove unanswered user message from history
            st.session_state.chat_history.pop()
