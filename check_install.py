import fitz          # PyMuPDF
import requests
import bs4           # BeautifulSoup
import pandas
import sentence_transformers
import faiss
import langchain
import openai
from dotenv import load_dotenv
import os

load_dotenv()

print("All libraries imported successfully!")
print(f"OpenAI key loaded: {'YES' if os.getenv('OPENAI_API_KEY') else 'NO — check your .env file'}")