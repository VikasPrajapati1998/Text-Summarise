# chatbot.py
import os
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from logger import setup_logger

os.environ["HF_HUB_OFFLINE"] = "0"

# ------------------------------------------------
# Initialize logger (writes to logs/<module>_timestamp.log)
# ------------------------------------------------
logger = setup_logger("gemmabot", log_dir="logs", level=logging.DEBUG)


class GemmaBot:
    """
    Object-oriented wrapper around your original prompt_ui script.
    """

    def __init__(self,
                 repo_id: str = "google/gemma-2-2b-it",
                 task: str = "conversational",
                 temperature: float = 0.7,
                 hf_env_var: str = "HUGGINGFACEHUB_API_TOKEN"):
        # Load environment variables
        load_dotenv(find_dotenv())
        self.api_token = os.getenv(hf_env_var)
        logger.debug(f"Hugging Face API Token loaded: {bool(self.api_token)}")

        if not self.api_token:
            logger.error(f"{hf_env_var} not found in environment variables")
            raise ValueError(f"{hf_env_var} is required")

        self.repo_id = repo_id
        self.task = task
        self.temperature = temperature

        try:
            logger.info(f"Initializing HuggingFaceEndpoint with model '{self.repo_id}' and task '{self.task}'")
            self.llm = HuggingFaceEndpoint(
                repo_id=self.repo_id,
                task=self.task,
                huggingfacehub_api_token=self.api_token,
                temperature=self.temperature
            )
        except Exception as e:
            logger.exception("Failed to initialize HuggingFaceEndpoint")
            raise

        try:
            logger.info("Initializing ChatHuggingFace model...")
            self.chat_model = ChatHuggingFace(llm=self.llm)
        except Exception as e:
            logger.exception("Failed to initialize ChatHuggingFace")
            raise

    def run(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        try:
            logger.info(f"Received query: {query[:80]}...")
            response = self.chat_model.invoke([HumanMessage(content=query)])
            content = getattr(response, "content", str(response))
            logger.info(f"Generated response: {content[:100]}...")
            return content
        except Exception as e:
            logger.exception("Error during model invocation")
            raise


# Example usage for standalone testing
if __name__ == "__main__":
    try:
        bot = GemmaBot()
        reply = bot.run("What is the capital of Nepal?")
        print(reply)
    except Exception as err:
        logger.exception(f"Failed to run GemmaBot: {err}")



