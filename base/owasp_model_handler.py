import os
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredExcelLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
from dotenv import load_dotenv
from .prompts import OWASP_TEMPLATE, OWASP_PROMPT
from source.source import REFERENCE

import warnings

warnings.filterwarnings("ignore")

load_dotenv(override=True)
BASE_URL = os.getenv("OPENROUTER_BASE_URL")
API_KEY = os.getenv("OPENROUTER_API_TOKEN")
MODEL = os.getenv("OPENROUTER_MODEL")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OWASPModelHandler:
    def __init__(self):
        self.llm = self.get_llm()
        self.embedding = self.get_embedding_model()
        self.vector_store = self.get_vector_store()
        self.chunking = self.get_chunking()

        self.template = OWASP_TEMPLATE
        self.prompt = OWASP_PROMPT
        self.reference = REFERENCE

        logger.info("Initialised RAG model handler")

        self.retriever = self.create_retriever(self.reference)
        self.rag_chain = self.create_rag_chain(self.retriever)

    def get_llm(self):
        return ChatOpenAI(
            model=MODEL,
            base_url=BASE_URL,
            api_key=API_KEY,
            temperature=1,
            max_completion_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def get_embedding_model(self):
        # https://huggingface.co/spaces/mteb/leaderboard
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
        )

    def get_vector_store(self):
        # https://python.langchain.com/docs/integrations/vectorstores/
        return FAISS  # Chroma

    def get_chunking(self):
        return RecursiveCharacterTextSplitter()

    def get_reranking_model(self):
        raise NotImplementedError

    def create_retriever(self, reference: str):
        if reference.endswith("xlsx"):
            logger.info("Using excel loader")
            loader = UnstructuredExcelLoader(reference)
        else:
            logger.info("Using web loader")
            loader = WebBaseLoader(reference)
        try:
            document = loader.load()
            document_chunks = self.chunking.split_documents(document)
            vector_store = self.vector_store.from_documents(
                document_chunks, self.embedding
            )
            retriever = vector_store.as_retriever()
            logger.info("Successfully created a retriever.")
            return retriever
        except Exception as e:
            logger.error(f"Failed to load from {reference}: {e}")

    def create_rag_chain(self, retriever):
        try:
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

            logger.info(f"RAG chain created: {rag_chain}")
            return rag_chain
        except Exception as e:
            logger.error(f"Failed to create rag chain: \n {e}")

    def query_model(self, user_input: str):
        response = self.rag_chain.invoke(user_input)
        return response

    async def query_model_async(self, user_input: str):
        response = await self.rag_chain.ainvoke(user_input)
        return response
