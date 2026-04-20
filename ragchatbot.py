import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# This simple function puts the document back together from chunks in the vector store.
# It's used in the langchain context.
def format_docs(docs):
    '''Takes a list of document chunks and formats them 
    into a single string for the question context.'''
    return "\n\n".join([doc.page_content for doc in docs])

# Create a barebones UI with streamlit
# Consists of a header with a sidebar
st.header("My First RAG Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload your PDF file and start asking questions", type="pdf")

# Extract contents from the pdf files and chunk it into smaller pieces
if file is not None:
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000, 
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    # Generate embeddings for the chunks of text
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small"
    )

    # store embeddings in vector store database.
    vector_store = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Ask a question about the document")

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4})
    
    llm = ChatOpenAI(
        model = "gpt-4o-mini",
        temperature = 0.3,
        max_tokens = 100
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are a helpful assistant answering questions about a PDF document.\n\n"
             "Guidelines:\n"
             "1. Provide complete, well-explained answers using the content below.\n"
             "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
             "3. If the contenxt mentions related information, include it to give fuller picture.\n"
             "4. Only use information from the provided context - do not ue outside knowledge.\n"
             "5. Summarize long information, ideally in bullets where needed\n"
             "6. If the informatioin is not in the context, say so politely.\n\n"
             "Context:\n{context}"),
            ("human", "{question}")
        ]
    )

    # Chaining everything together.
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough() }
        | prompt
        | llm
        | StrOutputParser()
    )

    # If there was a question, run the chain and display the response.
    if user_question:
        response = chain.invoke(user_question)
        st.write(response)