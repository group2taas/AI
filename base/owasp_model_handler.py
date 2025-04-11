import os
import re
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    UnstructuredExcelLoader,
    WebBaseLoader,
    RecursiveUrlLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field

from pprint import pprint
from typing import List
from typing_extensions import TypedDict
from loguru import logger
from dotenv import load_dotenv
from .prompts import OWASP_TEMPLATE, OWASP_PROMPT
from source.source import REFERENCE, ZAP_API_DOCS, WSTG_GUIDE

import warnings

warnings.filterwarnings("ignore")

load_dotenv(override=True)
BASE_URL = os.getenv("OPENROUTER_BASE_URL")
API_KEY = os.getenv("OPENROUTER_API_TOKEN")
MODEL = os.getenv("OPENROUTER_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_TOKEN")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OWASPModelHandler:
    def __init__(self):
        self.llm = self.get_llm()
        self.hallucination_grader = self.get_hallucination_grader()
        self.answer_grader = self.get_answer_grader()
        self.embedding = self.get_embedding_model()
        self.vector_store = self.get_vector_store()
        self.chunking = self.get_chunking()

        self.template = OWASP_TEMPLATE
        self.prompt = OWASP_PROMPT

        logger.info("Initialised RAG model handler")

        self.rag_chain = self.create_rag_chain()

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

    def get_hallucination_grader(self):
        llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeHallucinations)

        system = """You are a grader assessing whether the code in LLM generation can run without errors in a standard Python environment.\n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the code is expected to execute without error.\n
            Do not have to be too strict. Mainly check for syntax errors, missing imports, and other common issues.\n"""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "LLM generation: {generation}",
                ),
            ]
        )
        hallucination_grader = hallucination_prompt | structured_llm_grader
        logger.info("Initialised hallucination grader")
        return hallucination_grader

    def get_answer_grader(self):
        llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeAnswer)

        system = """You are a grader assessing whether the code in the LLM generation has sufficient test cases that are relevant to the user question. \n 
            Give a binary score 'yes' or 'no'. Yes' means that the code have sufficient and relevant test cases. Do not have to be too strict. \n"""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "User question: \n\n {question} \n\n LLM generation: {generation}",
                ),
            ]
        )
        answer_grader = answer_prompt | structured_llm_grader
        logger.info("Initialised answer grader")
        return answer_grader

    def get_chunking(self):
        # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        #     chunk_size=250, chunk_overlap=0
        # )
        text_splitter = RecursiveCharacterTextSplitter()
        return text_splitter

    def get_embedding_model(self):
        # https://huggingface.co/spaces/mteb/leaderboard
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
        )

    def get_vector_store(self):
        # https://python.langchain.com/docs/integrations/vectorstores/
        return FAISS  # Chroma

    def get_reranker(self):
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=model, top_n=3)
        return compressor

    def create_retriever(self, document_chunks):

        logger.info("Creating retrievers...")
        retriever1 = BM25Retriever.from_documents(document_chunks)
        vector_store = self.get_vector_store()
        retriever2 = vector_store.from_documents(
            document_chunks,
            self.embedding,
        ).as_retriever()
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever1, retriever2],
            weights=[0.5, 0.5],
        )
        compressor = self.get_reranker()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )
        return compression_retriever

    def _get_all_links(self, url):
        logger.info(f"Getting all sublinks from {url}")
        loader = RecursiveUrlLoader(
            url, prevent_outside=True, base_url=url, exclude_dirs=[]
        )
        docs = loader.load()
        links = [docs.metadata["source"] for docs in docs]
        return links

    def load_documents(self):

        web_links = self._get_all_links(WSTG_GUIDE)
        web_links.append(ZAP_API_DOCS)

        logger.info("Loading web documents...")
        web_loader = WebBaseLoader(web_links)
        web_loader.requests_per_second = 2
        web_documents = web_loader.aload()

        logger.info("Loading excel documents...")
        excel_loader = UnstructuredExcelLoader(REFERENCE)
        excel_documents = excel_loader.load()

        documents = web_documents + excel_documents

        return documents

    def create_rag_chain(self):
        try:
            documents = self.load_documents()
            document_chunks = self.chunking.split_documents(documents)
            retriever = self.create_retriever(document_chunks)
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

    def _parse_output(self, raw_output):
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", raw_output, flags=re.DOTALL)
        if match:
            code = match.group(1)
        else:
            code = raw_output
        return code.strip()

    async def generate(self, state):
        print("---GENERATE---")
        question = state["question"]

        # RAG generation
        generation = await self.rag_chain.ainvoke(question)
        return {"question": question, "generation": generation}

    async def grade_generation_v_documents_and_question(self, state):

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        generation = state["generation"]

        code = self._parse_output(generation)
        print(code)
        score = await self.hallucination_grader.ainvoke({"generation": code})
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            print("---DECISION: TESTING SCRIPT SEEMS TO WORK---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION/DOCUMENT---")
            score = await self.answer_grader.ainvoke(
                {"question": question, "generation": code}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: TESTING SCRIPT HAVE GOOD COVERAGE---")
                return "ok"
            else:
                print("---DECISION: TESTING SCRIPT DOES NOT HAVE GOOD COVERAGE---")
                return "not ok"
        else:
            print("---DECISION: TESTING SCRIPT DOES NOT SEEM TO WORK, RE-TRY---")
            return "useless"

    def construct_graph(self):
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("generate", self.generate)

        # Build graph
        workflow.add_edge(START, "generate")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "useless": "generate",
                "ok": END,
                "not ok": "generate",
            },
        )

        graph = workflow.compile()
        logger.info(f"Constructed graph: \n {graph.get_graph()}")
        return graph

    # def query_model(self, user_input: str):
    #     response = self.rag_chain.invoke(user_input)
    #     return response

    async def query_model_async(self, user_input: str):
        # response = await self.rag_chain.ainvoke(user_input)
        graph = self.construct_graph()
        inputs = {"question": user_input}

        async for output in graph.astream(inputs):
            for key, value in output.items():
                pprint(f"Node '{key}':")

        response = value["generation"]
        return response


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Generated testing script can run, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Generated testing script have sufficient and relevant coverage, 'yes' or 'no'"
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
    """

    question: str
    generation: str
