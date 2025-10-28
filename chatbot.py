# prompt_ui.py
import logging
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

os.environ["HF_HUB_OFFLINE"] = "0"

# Configure logging (same style as your original script)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GemmaBot:
    """
    Object-oriented wrapper around your original prompt_ui script.

    Usage:
        client = GemmaBot()
        response_text = client.run("What is the capital of India?")
    """

    def __init__(self,
                 repo_id: str = "google/gemma-2-2b-it",
                 task: str = "conversational",
                 temperature: float = 0.7,
                 hf_env_var: str = "HUGGINGFACEHUB_API_TOKEN"):
        """
        Initialize GemmaBot: loads env, checks token, and initializes HF endpoint + chat model.
        This mirrors the behavior of your original procedural script.
        """
        # Load environment variables
        load_dotenv(find_dotenv())
        self.api_token = os.getenv(hf_env_var)
        logger.debug(f"Hugging Face API Token: {self.api_token}")

        if not self.api_token:
            logger.error(f"{hf_env_var} not found in environment variables")
            raise ValueError(f"{hf_env_var} is required")

        self.repo_id = repo_id
        self.task = task
        self.temperature = temperature

        # Initialize model objects (same as original)
        try:
            logger.info(f"Initializing HuggingFaceEndpoint with model '{self.repo_id}' and task '{self.task}'")
            self.llm = HuggingFaceEndpoint(
                repo_id=self.repo_id,
                task=self.task,
                huggingfacehub_api_token=self.api_token,
                temperature=self.temperature
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFaceEndpoint: {e}")
            raise

        try:
            logger.info("Initializing ChatHuggingFace")
            self.chat_model = ChatHuggingFace(llm=self.llm)
        except Exception as e:
            logger.error(f"Failed to initialize ChatHuggingFace: {e}")
            raise

    def run(self, query: str) -> str:
        """
        Send the given query string to the chat model and return the response string.

        Args:
            query: the prompt to send

        Returns:
            response content as string

        Raises:
            ValueError: if query is empty
            Exception: if the model invocation fails
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        try:
            logger.info(f"Sending query: {query}")
            response = self.chat_model.invoke([HumanMessage(content=query)])
            # try to get .content (keeps same assumption as your original code)
            content = getattr(response, "content", None)
            if content is None:
                # fallback to string representation
                content = str(response)
            logger.info(f"Received response: {content}")
            return content
        except Exception as e:
            logger.error(f"Error during model invocation: {e}")
            raise


# Example usage: keeps same behavior as your original script when run directly.
if __name__ == "__main__":
    # Example query (same as your original)
    example_query = "What is the capital of Nepal?"

    try:
        client = GemmaBot()
        resp = client.run(example_query)
        print(resp)
    except Exception as error:
        logger.error(f"Failed to run GemmaBot: {error}")
        raise
