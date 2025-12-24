import json
from typing import Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from datetime import datetime

from app.core.config import settings
from app.services.rag_service import retrieve_context


class IntentClassification(BaseModel):
    intent: Literal["BOOKING", "INFO", "GENERAL"] = Field(
        description="The user's intent: BOOKING (scheduling/availability), INFO (questions about services/prices/location), or GENERAL (greetings/small talk)."
    )


class BookingDetails(BaseModel):
    name: Optional[str] = Field(description="The user's name if provided, else null.")
    service: Optional[str] = Field(
        description="The service the user wants to book if provided, else null."
    )
    time: Optional[str] = Field(
        description="The preferred time for the appointment if provided, else null."
    )
    phone: Optional[str] = Field(
        description="The user's phone number if provided, else null."
    )



class AgentState(TypedDict):
    messages: List[BaseMessage]
    context: str
    intent: str
    booking_info: Dict[str, str]


def get_llm():
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")
    return ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=settings.OPENAI_API_KEY)


def classify_intent(state: AgentState):
    llm = get_llm()
    messages = state["messages"]
    last_message = messages[-1].content
    booking_info = state.get("booking_info", {})

    print(f"Classifying intent for message: '{last_message}'")
    print(f"Current booking info: {booking_info}")

    structured_llm = llm.with_structured_output(IntentClassification)

    prompt = f"""
    Classify the user's intent based on their last message and current context.
    
    Current Booking Info: {json.dumps(booking_info)}
    
    User message: {last_message}
    
    Note: If the user is providing details (like name, time, phone) to complete a booking, classify as BOOKING.
    """

    result = structured_llm.invoke(prompt)
    intent = result.intent if result else "GENERAL"
    
    print(f"Intent classified as: {intent}")

    return {"intent": intent}


def retrieve_node(state: AgentState):
    query = state["messages"][-1].content
    print(f"Retrieving context for query: '{query}'")
    context = retrieve_context(query)
    print(f"Retrieved context length: {len(context)} characters")
    return {"context": context}


def booking_node(state: AgentState):
    llm = get_llm()
    messages = state["messages"]
    booking_info = state.get("booking_info", {}) or {}

    print(f"Processing booking node")
    print(f"Current booking info: {booking_info}")

    structured_llm = llm.with_structured_output(BookingDetails)

    extraction_prompt = f"""
    You are managing a booking simulation for Elite Body Home Clinic.
    Current Time: {datetime.now()}
    Current Booking Info: {json.dumps(booking_info)}
    User History: {[m.content for m in messages if isinstance(m, HumanMessage)]}
    
    Extract any new booking information from the conversation.
    """

    try:
        result = structured_llm.invoke(extraction_prompt)
        if result:
            new_info = result.model_dump(exclude_none=True)
            for k, v in new_info.items():
                if v:
                    booking_info[k] = v
            print(f"Extracted new booking info: {new_info}")
    except Exception as e:
        print(f"Booking extraction error: {e}")

    missing = [
        feild
        for feild in ["name", "service", "time", "phone"]
        if not booking_info.get(feild)
    ]

    print(f"Missing booking fields: {missing}")

    if not missing:
        print(f"Booking completed successfully!")
        response_text = (
            f"Thank you {booking_info.get('name')}. I have simulated a booking for {booking_info.get('service')} "
            f"at {booking_info.get('time')}. We will contact you at {booking_info.get('phone')} to confirm. "
            f"{settings.DISCLAIMER}"
        )
        return {"messages": [AIMessage(content=response_text)], "booking_info": booking_info}

    print(f"Need to collect missing information")
    prompt_missing = f"""
    You are a receptionist at Elite Body Home Clinic.
    The user wants to book an appointment.
    We have the following info: {json.dumps(booking_info)}
    We are missing: {', '.join(missing)}

    Politely ask the user for the missing details.
    """
    response = llm.invoke(prompt_missing)
    print(f"Asked user for missing info: {response.content}")
    return {"messages": [response], "booking_info": booking_info}


def general_response_node(state: AgentState):
    llm = get_llm()
    messages = state["messages"]
    context = state.get("context", "")

    print(f"Generating general response")
    print(f"Context available: {'Yes' if context else 'No'}")
    if context:
        print(f"Context length: {len(context)} characters")

    system_prompt = """You are a professional, empathetic AI assistant for Elite Body Home Clinic.
    Your tone should be healthcare-professional yet warm.
    Use the provided context to answer questions accurately.
    If the context doesn't contain the answer, politely say you don't have that information and suggest contacting the clinic directly.
    Do not hallucinate facts.
    """

    messages_payload = [SystemMessage(content=system_prompt)]
    if context:
        messages_payload.append(SystemMessage(content=f"Relevant Context:\n{context}"))

    messages_payload.extend(messages)

    response = llm.invoke(messages_payload)
    print(f"Generated response: {response.content}")
    return {"messages": [response]}


workflow = StateGraph(AgentState)

workflow.add_node("classify", classify_intent)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("booking", booking_node)
workflow.add_node("respond", general_response_node)

workflow.set_entry_point("classify")


def route_intent(state):
    intent = state["intent"]
    if intent == "BOOKING":
        return "booking"
    elif intent == "INFO":
        return "retrieve"
    else:
        return "respond"


workflow.add_conditional_edges("classify", route_intent)
workflow.add_edge("retrieve", "respond")
workflow.add_edge("booking", END)
workflow.add_edge("respond", END)

chat_app = workflow.compile()
