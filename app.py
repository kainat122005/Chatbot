import os
import streamlit as st
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain.chains import RetrievalQA

# File upload
uploaded_file = st.file_uploader("Upload the document here", type="docx")
if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = Docx2txtLoader(uploaded_file.name)
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Embedding
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Qdrant setup
    qdrant_url = "https://fe58f34e-8a11-44b7-bc37-b36c7b67f516.us-west-1-0.aws.cloud.qdrant.io:6333"
    qdrant_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ZOuPanOWtPTZX6-ixCgGJ-SytMMUBco320lUIenAOgk"
    collection_name = "hope_cluster_v2"

    qdrant = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        recreate_collection=True 
    )

    # Retrievel
    st.title("Chatbot")
    st.subheader("Ask any question from uploaded document.")
    query = st.text_input("Ask any question")
    if query:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key="AIzaSyAaI6cEtck9zu9Vb0UphPTez2BkFRzFXdw"
        )

        retrieval = qdrant.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retrieval)
        result = qa_chain.run(query)
        st.write("Answer:", result)
