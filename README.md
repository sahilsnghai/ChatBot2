# Elite Body Home Clinic Chatbot

This project implements an AI-powered chatbot for Elite Body Home Clinic using FastAPI, LangChain, LangGraph, and ChromaDB.

## Features

- **RAG Architecture**: Retrieves information from the clinic's website content.
- **Intent Classification**: Distinguishes between Booking, Info, and General queries.
- **Booking Simulation**: Simulates an appointment booking flow.
- **Context Awareness**: Maintains session history.

## Setup

1. **Install Dependencies**:

   ```bash
   uv sync
   ```

2. **Environment Variables**:
   Create a `.env` file in the root directory and add your OpenAI API Key:

   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

3. **Data Ingestion**:
   Run the ingestion script to populate the vector database:

   ```bash
   python ingestion/ingest.py
   ```

4. **Run the Application**:
   ```bash
   uvicorn main:app --reload
   ```

## API Usage

Endpoint: `POST /api/chat`

Request Body:

```json
{
  "message": "I want to book a botox appointment",
  "session_id": "user123"
}
```

Response:

```json
{
  "response": "I can help you with that. What time would you prefer?",
  "booking_info": {
    "service": "Botox"
  }
}
```

## Project Structure

- `app/`: Application source code.
  - `api/`: FastAPI routes.
  - `core/`: Configuration.
  - `services/`: Business logic (Chat, RAG).
- `data/`: Data storage (ChromaDB, text content).
- `ingestion/`: Scripts for data ingestion.
- `tests/`: Tests.
