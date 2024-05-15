# Streamlit + Google Gemini API + LangChain

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

with st.sidebar:
    "[홈으로 돌아가기](http://localhost:8080/)"
    gemini_api_key = st.text_input("Gemini API Key", key="chatbot_api_key", type="password")
    uploaded_file = st.file_uploader("PDF 파일 업로드", type=("pdf"))
    process = st.button("랭체인 연동하기")

if process:
    # session_state 초기화
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    # PDF 파일 처리
    file_name = uploaded_file.name
    with open(file_name, "wb") as file:
        file.write(uploaded_file.getvalue())

    # PDF 정보 가져오기
    loader = PyPDFLoader(file_name)
    pages = loader.load_and_split()

    # PDF 내용을 작은 chunk 단위로 나누기
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(pages)

    # 임베딩 모델 설정
    model_name = "jhgan/ko-sroberta-multitask" # (KorNLU 데이터셋에 학습시킨 한국어 임베딩 모델)
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # ChromaDB 저장 & Retriever 설정
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    vectorstore_retriever = vectorstore.as_retriever()

    st.session_state.retriever = vectorstore_retriever


st.title("🚀 PDF 파일처리 서비스 💬")
st.caption("Streamlit + Google Gemini API + LangChain 를 연동했습니다.")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "PDF 파일을 업로드한 후, PDF에서 궁금한 점을 질문해주세요."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not gemini_api_key:
        st.info("Google Gemini API 키를 입력하고 서비스를 이용해주세요.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # LLM 설정
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=gemini_api_key)
    
    # 프롬프트 템플릿 설정
    template = """다음과 같은 맥락을 사용하여 마지막 질문에 대답하십시오.
    # 답변은 최대 세 문장으로 하고 가능한 한 간결하게 유지하십시오.
    # {context}
    # 질문: {question}
    # 도움이 되는 답변:"""
    rag_prompt_custom = PromptTemplate.from_template(template)

    # RAG chain 설정
    rag_chain = {"context": st.session_state.retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm
    response = rag_chain.invoke(f'{prompt}')
    msg = response.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)