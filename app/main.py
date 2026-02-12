from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.chat import ChatRequest, ChatResponse
from app.core.rag_chain import rag_chain

app = FastAPI(
    title="법률 RAG 챗봇 API",
    description="한국 조달 법률 문서 기반 RAG 질의응답 챗봇",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"message": "Hello World"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = rag_chain.invoke(
        {"input": req.question},
        config={"configurable": {"session_id": req.session_id}},
    )
    return ChatResponse(session_id=req.session_id, answer=answer)
