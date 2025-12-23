from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.core.kv_store import session_store
from app.services.chat_service import chat_app
from datetime import datetime

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str
    booking_info: Optional[Dict[str, str]] = {}

class BookingInfo(BaseModel):
    name: Optional[str]
    service: Optional[str]
    phone: Optional[str]
    time: Optional[datetime]

class ChatResponse(BaseModel):
    response: str
    booking_info: BookingInfo


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    print(f"🔄 Processing chat request for session: {session_id}")
    print(f"📝 User message: {request.message}")
    print(f"📞 Booking info received: {request.booking_info}")

    existing_session = session_store.mget([session_id])[0]

    if not existing_session:
        print(f"🆕 Creating new session for {session_id}")
        session_data = {"messages": [], "booking_info": {}}
    else:
        print(f"📚 Loading existing session for {session_id}")
        session_data = existing_session

    if request.booking_info:
        print(f"📝 Updating booking info with: {request.booking_info}")
        session_data["booking_info"].update(request.booking_info)

    session_data["messages"].append(HumanMessage(content=request.message))
    print(f"💬 Total messages in session: {len(session_data['messages'])}")

    try:
        print("🤖 Invoking chat application...")
        inputs = {
            "messages": session_data["messages"],
            "booking_info": session_data["booking_info"],
            "intent": "",
            "context": "",
        }

        result = chat_app.invoke(inputs)
        print(f"✅ Chat application completed successfully")

        session_data["messages"] = result["messages"]
        session_data["booking_info"] = result.get("booking_info", {})

        session_store.mset([(session_id, session_data)])
        print(f"💾 Session data saved to store")

        last_message = result["messages"][-1].content
        print(f"📤 Response generated: {last_message[:50]}...")

        return ChatResponse(
            response=last_message, booking_info=session_data["booking_info"]
        )
    except Exception as e:
        print(f"❌ Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
