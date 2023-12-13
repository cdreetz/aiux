import os
from dotenv import load_dotenv
import openai
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

class OpenAIClient:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            cls._instance = openai.OpenAI(api_key=api_key)
        return cls._instance

class ChromaDBClient:
    _instances = {}

    @classmethod
    def get_instance(cls, environment='dev'):
        if environment not in cls._instances:
            if environment == 'dev':
                host = os.environ.get("DEV_CHROMADB_HOST")
            elif environment == 'test':
                host = os.environ.get("TEST_CHROMADB_HOST")
            else:
                raise ValueError("Invalid environment specified")
            
            cls._instances[environment] = chromadb.HttpClient(host=host, port=8000)
        return cls._instances[environment]
    
    @classmethod
    def setup_chroma_collection(cls, environment='dev'):
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key = os.environ.get("OPENAI_API_KEY"),
            model_name = "text-embedding-ada-002"
        )
        chroma_client = cls.get_instance(environment)
        return chroma_client.get_collection(name=os.environ.get("COLLECTION_NAME"), embedding_function=openai_ef)
