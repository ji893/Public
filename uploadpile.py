import os
import tempfile
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

#  OpenAI API Key 설정
load_dotenv()

# Streamlit UI 구성
st.set_page_config(page_title="파일 업로드 + 헌법 Q&A 챗봇", layout="centered")
                                                                            
st.header(" 업로드된 문서 기반 Q&A 챗봇 ")

# GPT 모델 선택
selected_model = st.selectbox("사용할 GPT 모델을 선택하세요:", ("gpt-4o", "gpt-3.5-turbo-0125"))

# PDF 업로드
uploaded_file = st.file_uploader("📎 PDF 파일을 업로드하세요", type=["pdf"])

# 🔑 PDF 해시 생성 함수
def get_file_hash(file) -> str:  
    content = file.read()
    file.seek(0) 
    return hashlib.md5(content).hexdigest()

# PDF 로드 및 분할

@st.cache_resource 
def load_and_split_pdf(file) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name
    loader = PyPDFLoader(tmp_file_path)
    return loader.load_and_split()

# FAISS 저장/로드 통합 함수
@st.cache_resource  
def load_or_create_vectorstore(_docs, file_hash):  
    index_path = os.path.join("faiss_index", file_hash) 
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(index_path):  
        return FAISS.load_local(index_path, embedding_model) 
    # 없다면 새로 생성
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0) 
    split_docs = text_splitter.split_documents(_docs)
    for doc in split_docs:
        doc.metadata["source"] = f"{doc.metadata.get('source', '업로드 파일')} (p.{doc.metadata.get('page', 'n/a')})" 
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    os.makedirs("faiss_index", exist_ok=True) 
    vectorstore.save_local(index_path) 
    return vectorstore

# RAG 체인 구성
def initialize_rag_chain(docs, file_hash, selected_model):
    vectorstore = load_or_create_vectorstore(docs, file_hash)
    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and a new question, return a standalone version of the question."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say you don't know. 
Use polite Korean and include emoji.\n\n{context}"""),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# 
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "업로드한 문서에 대해 궁금한 것을 질문해 주세요 😊"}]

# 
if uploaded_file:
    file_hash = get_file_hash(uploaded_file) 
    
    with st.spinner("PDF 분석 중..."): 
        pages = load_and_split_pdf(uploaded_file) 
        
        rag_chain = initialize_rag_chain(pages, file_hash, selected_model)
        
        chat_history = StreamlitChatMessageHistory(key="chat_messages")
        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )


    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input("질문을 입력하세요"):
        st.chat_message("human").write(prompt)
        with st.chat_message("ai"):
            with st.spinner("답변 생성 중..."):
                config = {"configurable": {"session_id": "upload_session"}}
                response = conversational_chain.invoke({"input": prompt}, config)
                answer = response["answer"]
                st.write(answer)

                with st.expander(" 참고한 문서 보기"):
                    for doc in response.get("context", []):
                        st.markdown(f"{doc.metadata.get('source', '알 수 없음')}", help=doc.page_content)


