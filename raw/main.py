from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model,
)

print("Database count:", vectorstore._collection.count())

retriever = vectorstore.as_retriever(
    search_type="mmr",  # maximum marginal relevance. finds answer that is most relevant to the user's question.
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.7,
    },  # reduces the number of chunks passed to the LLM.
)

# Calling LLM Model
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are ContextAI, a friendly, funny, and intelligent PDF AI assistant. Always introduce yourself to buld a friendly relationship.
        
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
        11. For unrelated factual questions, explain that you are a PDF-based assistant.
        12. Use conversation history to understand follow-up questions, but answer factual questions only from the provided PDF context.
        """,
        ),
        (
            "human",
            """
        Context:
        {context}

        Question
        {question}

        Provide the answer with its PDF page reference.
        """,
        ),
    ]
)


print("=" * 10, "RAG Chatbot", "=" * 10)

print("press # to exit")

while True:
    query = input("You: ")
    if query == "#":
        break

    docs = retriever.invoke(query)
    print("\n========== RETRIEVED DOCUMENTS ==========")

    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:1000])

    print("\n========== END DOCUMENTS ==========")

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({"context": context, "question": query})

    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")
