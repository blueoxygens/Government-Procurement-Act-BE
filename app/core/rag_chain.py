import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma")
COLLECTION_NAME = "law-docs"

# 세션별 대화 이력 저장소 (인메모리)
_session_histories: dict[str, ChatMessageHistory] = {}


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_histories:
        _session_histories[session_id] = ChatMessageHistory()
    return _session_histories[session_id]


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    database = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embedding,
    )

    retriever = database.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "당신은 한국 법률 전문가입니다. "
            "아래 제공된 법률 문서 내용을 참고하여 사용자의 질문에 정확하게 답변해주세요. "
            "문서에 없는 내용은 답변하지 마세요.\n\n"
            "[참고 문서]\n{context}",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    def retrieve_and_build(input_dict):
        question = input_dict["input"]
        docs = retriever.invoke(question)
        return {
            "context": _format_docs(docs),
            "input": question,
            "history": input_dict.get("history", []),
        }

    rag_chain = (
        RunnableLambda(retrieve_and_build)
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return chain_with_history


# 싱글톤 체인 인스턴스
rag_chain = build_rag_chain()
