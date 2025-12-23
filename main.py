from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.routes import router as chat_router

load_dotenv()

app = FastAPI(title="Elite Body Home Clinic Chatbot")

app.include_router(chat_router, prefix="/api")
print("Chat router included successfully")


@app.get("/")
def read_root():
    print("Root endpoint accessed")
    return {"message": "Welcome to Elite Body Home Clinic Chatbot API"}
