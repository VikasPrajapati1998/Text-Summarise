# prompt_ui.py
import logging
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
import streamlit as st

os.environ["HF_HUB_OFFLINE"] = "0"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(find_dotenv())
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
logger.debug(f"Hugging Face API Token: {api_token}")

if not api_token:
    logger.error("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is required")

# Initialize the Hugging Face Endpoint
try:
    logger.info(f"Initializing HuggingFaceEndpoint with model 'google/gemma-2-2b-it' and task 'conversational'")
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-2b-it",
        task="conversational",
        huggingfacehub_api_token=api_token,
        temperature=0.7
    )
except Exception as e:
    logger.error(f"Failed to initialize HuggingFaceEndpoint: {str(e)}")
    raise

# Initialize ChatHuggingFace for conversational tasks
try:
    logger.info("Initializing ChatHuggingFace")
    chat_model = ChatHuggingFace(llm=llm)
except Exception as e:
    logger.error(f"Failed to initialize ChatHuggingFace: {str(e)}")
    raise

# Query the model
query = "What is the capital of India?"
try:
    logger.info(f"Sending query: {query}")
    response = chat_model.invoke([HumanMessage(content=query)])
    logger.info(f"Received response: {response.content}")
    print(response.content)
except Exception as e:
    logger.error(f"Error during model invocation: {str(e)}")
    raise

