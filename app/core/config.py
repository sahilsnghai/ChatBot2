import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CHROMA_DB_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data/chroma_db"
    )
    COLLECTION_NAME = "elite_body_home"
    DISCLAIMER = "(Disclaimer: This is a simulation, no actual appointment has been booked)."



settings = Settings()
