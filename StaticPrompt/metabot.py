# metabot.py
import os
import logging
import json
import time
from dotenv import load_dotenv, find_dotenv

# Logger setup (fallback if logger.py missing)
try:
    from logger import setup_logger
    logger = setup_logger("metabot", log_dir="logs", level=logging.DEBUG)
except Exception:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    logger = logging.getLogger("metabot")
    os.makedirs("logs", exist_ok=True)

# LangChain imports (with fallback flag)
_HAS_LANGCHAIN = False
try:
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    from langchain_core.messages import HumanMessage, SystemMessage
    _HAS_LANGCHAIN = True
    logger.info("LangChain integration available.")
except ImportError:
    logger.warning("LangChain not available; using REST fallback only.")

# Standard libs for REST
import requests

os.environ.setdefault("HF_HUB_OFFLINE", "0")


class MetaBot:  # Renamed to LlamaBot? Nah, keeping for compatibility—it's a generic HF wrapper now
    """
    HF Inference API wrapper for Llama 3.1-8B-Instruct.
    Prefers LangChain ChatHuggingFace for conversational tasks;
    Falls back to raw REST with chat payload.
    """

    def __init__(
        self,
        repo_id: str = "meta-llama/Llama-3.1-8B-Instruct",  # <-- Your requested model
        task: str = "conversational",  # <-- Native support for Llama
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        hf_env_var: str = "HUGGINGFACEHUB_API_TOKEN",
        timeout: int = 60,
    ):
        load_dotenv(find_dotenv())
        self.api_token = os.getenv(hf_env_var)
        logger.debug(f"HF API Token loaded: {bool(self.api_token)}")

        if not self.api_token:
            logger.error(f"{hf_env_var} not found.")
            raise ValueError(f"{hf_env_var} is required. Add to .env file.")

        self.repo_id = repo_id
        self.task = task
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.timeout = int(timeout)

        # LangChain init (if available)
        self.llm = None
        self.chat_model = None
        if _HAS_LANGCHAIN:
            try:
                logger.info(f"Initializing HuggingFaceEndpoint: {repo_id} (task: {task})")
                self.llm = HuggingFaceEndpoint(
                    repo_id=self.repo_id,
                    task=self.task,
                    huggingfacehub_api_token=self.api_token,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                )
                logger.info("Initializing ChatHuggingFace...")
                self.chat_model = ChatHuggingFace(llm=self.llm)
                logger.info("LangChain chat integration ready.")
            except Exception as e:
                logger.exception("LangChain init failed; using REST fallback.")
                self.llm = None
                self.chat_model = None

        # REST setup
        self.rest_url = f"https://api-inference.huggingface.co/models/{self.repo_id}"
        self.rest_headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _call_langchain(self, query: str) -> str:
        """Invoke via ChatHuggingFace (OpenAI-style messages)."""
        if not self.chat_model:
            raise RuntimeError("Chat model not available.")

        try:
            system_msg = SystemMessage(content="You are a helpful AI assistant focused on Generative AI and Machine Learning.")
            logger.debug("Invoking ChatHuggingFace...")
            response = self.chat_model.invoke([system_msg, HumanMessage(content=query)])
            content = getattr(response, "content", str(response))
            return content.strip()
        except Exception as e:
            logger.exception("LangChain invocation failed.")
            raise

    def _call_rest_conversational(self, query: str) -> str:
        """REST call for conversational task (messages format)."""
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant focused on Generative AI and Machine Learning."},
                {"role": "user", "content": query},
            ],
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            },
        }

        logger.debug(f"REST payload: {json.dumps(payload, indent=2)}")
        try:
            resp = requests.post(self.rest_url, headers=self.rest_headers, json=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as re:
            logger.exception("Network error.")
            raise

        if resp.status_code == 200:
            try:
                data = resp.json()
            except json.JSONDecodeError:
                logger.error("Invalid JSON from HF.")
                raise RuntimeError("Invalid response from HF API.")

            # Llama conversational returns list of dicts with 'generated_text'
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(f"HF Error: {data['error']}")
            logger.warning(f"Unexpected response: {data}")
            return json.dumps(data)

        else:
            err_text = resp.json() if resp.headers.get('content-type') == 'application/json' else resp.text
            logger.error(f"HF API error {resp.status_code}: {err_text}")
            if resp.status_code == 401:
                raise RuntimeError("401 Unauthorized - Invalid token.")
            if resp.status_code == 403:
                raise RuntimeError("403 Forbidden - Accept model license on HF or token lacks access.")
            if resp.status_code == 429:
                raise RuntimeError("429 Rate limited - Wait and retry.")
            raise RuntimeError(f"HF API failed: {resp.status_code} - {err_text}")

    def run(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        logger.info(f"Query: {query[:100]}...")

        # Try LangChain first
        if _HAS_LANGCHAIN and self.chat_model:
            try:
                return self._call_langchain(query)
            except Exception as e:
                logger.warning(f"LangChain failed: {e}; falling back to REST.")

        # REST fallback
        return self._call_rest_conversational(query)


# Standalone test
if __name__ == "__main__":
    try:
        bot = MetaBot(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="conversational",
            temperature=0.7,
            max_new_tokens=256,
        )
        q = "What is the capital of Nepal?"
        logger.info(f"Test query: {q}")
        reply = bot.run(q)
        print("---- MODEL REPLY ----")
        print(reply)
    except Exception as err:
        logger.exception("Test failed.")
        print(f"ERROR: {err}")