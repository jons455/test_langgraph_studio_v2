import httpx

import getpass
import os

from langchain_openai import ChatOpenAI


from dotenv import load_dotenv

load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

API_URL = os.getenv("SIEMENS_API_ENDPOINT")
API_KEY = os.getenv("SIEMENS_API_KEY")

llm = ChatOpenAI( model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,)

llm.invoke("Hello how are you?")