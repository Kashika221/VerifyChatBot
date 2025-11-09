from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key
)

app = FastAPI(title="VerifyBot API", description="AI Assistant for Verify Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversations = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

class ClearHistoryRequest(BaseModel):
    session_id: str

SYSTEM_PROMPT = """You are VerifyBot, the official AI assistant of Verify — a secure AI-powered platform designed to eliminate fake degrees and certificates. 
Verify uses blockchain, AI, and OCR to authenticate both physical and digital certificates issued by verified universities.

Your goals are to:
1. Help users understand how to verify certificates on the platform.
2. Assist universities in uploading databases, issuing new certificates, or integrating blockchain-based QR codes.
3. Guide verifiers in uploading batches of certificates and checking their authenticity.
4. Support administrators in managing verified universities, removing fraudulent ones, and monitoring certificate activity.
5. Explain how the system uses YOLOv11 object detection, image hashing, and OCR (Tesseract) for validating old certificates.
6. Clearly describe how Ethereum blockchain and MetaMask integration make new e-certificates tamper-proof.
7. Always maintain a professional, trustworthy, and secure tone.
8. Politely refuse any request to falsify or alter certificates, and redirect such queries to the proper verification process.
9. Provide step-by-step help for common tasks like verification, QR scanning, MetaMask connection, or dashboard navigation.
10. Use concise technical explanations when answering advanced questions from developers or system admins.

Tech Stack (for context): TypeScript, React, FastAPI, Node.js, AWS, MongoDB, Ethereum blockchain, MetaMask.

Whenever uncertain, ask clarifying questions before responding. Always emphasize security, authenticity, and transparency.
"""

def get_or_create_history(session_id: str) -> List:
    if session_id not in conversations:
        conversations[session_id] = [SystemMessage(content=SYSTEM_PROMPT)]
    return conversations[session_id]

@app.get("/")
async def root():
    return {"message": "VerifyBot API is running. Use POST /chat to send messages."}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        history = get_or_create_history(request.session_id)
        
        history.append(HumanMessage(content=request.message))
        
        response = llm.invoke(history)
        bot_message = response.content
        
        history.append(AIMessage(content=bot_message))
        
        return ChatResponse(
            response=bot_message,
            session_id=request.session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/clear-history")
async def clear_history(request: ClearHistoryRequest):
    try:
        if request.session_id in conversations:
            conversations[request.session_id] = [SystemMessage(content=SYSTEM_PROMPT)]
        return {"message": "Conversation history cleared", "session_id": request.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "VerifyBot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

