# app.py
# Streamlit RAG (FAISS + LLM)

import streamlit as st
from typing import List

# LangChain components
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Streamlit UI setup
st.set_page_config(page_title="Azure RAG Assistant", layout="wide")
st.title("📄 Azure-Deployed RAG Assistant")
st.caption("LLMs + Prompt Engineering + FAISS + Azure")

# Prompt (this is prompt engineering)
SYSTEM_PROMPT = """
You are an expert assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say:
"I don't know based on the provided documents."
Cite the source document at the end of your answer.
"""

# Helper functions
@st.cache_resource
def load_vectorstore(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def ask_llm(question: str, context: str) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini"  # or whichever model you are using
    )

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}
"""

    response = llm.predict(prompt)
    return response

# Sidebar - document upload
with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload text files (.txt)",
        type=["txt"],
        accept_multiple_files=True
    )


# Load documents
docs = []

if uploaded_files:
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": file.name}
            )
        )


# Main app logic
if docs:
    vectorstore = load_vectorstore(docs)
    st.success("Documents indexed successfully!")

    query = st.text_input("Ask a question about your documents")

    if query:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(query)

        context = "\n\n".join(
            [
                f"Source: {d.metadata['source']}\n{d.page_content}"
                for d in relevant_docs
            ]
        )

        answer = ask_llm(query, context)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            st.text(context)
else:
    st.info("Upload at least one document to begin.")
