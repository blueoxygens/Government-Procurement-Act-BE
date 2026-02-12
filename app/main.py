from fastapi import FastAPI

from app.models.chat import ChatRequest, ChatResponse
from app.core.rag_chain import rag_chain

app = FastAPI()


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
