import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    FactualCorrectness,
    SimpleCriteriaScore,
    RubricsScore,
)
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from source.source import REFERENCE

BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.append(str(BASE_DIR))

load_dotenv(override=True)
BASE_URL = os.getenv("OPENROUTER_BASE_URL")
API_KEY = os.getenv("OPENROUTER_API_TOKEN")
MODEL = os.getenv("OPENROUTER_MODEL")

OPENAI_API_KEY = os.getenv("OPENAI_API_TOKEN")


def filter_json(result: List):
    output = dict()
    for data in result:
        if data.get("type", "") == "result":
            url = data.get("target_url", "")
            test = data.get("test_case", "")
            res = data.get("result", "")
            alerts = data.get("alert_details", [])
            if url not in output:
                output[url] = []
            output[url].append(test)
            if isinstance(res, str) and not res.lower().startswith("error"):
                output[url].append(res)

            for alert in alerts:
                alert_url = alert.get("url", "")
                alert_desc = alert.get("description", "")
                alert_risk = alert.get("risk", "")
                if alert_url not in output:
                    output[alert_url] = []
                output[alert_url].append(f"{alert_desc} ({alert_risk} risk)")

    return output


def run_eval_with_ai(results: Dict):
    # with open(f"{BASE_DIR}/output/test_results.json", "r") as f:
    #     results = json.load(f)
    with open(f"{BASE_DIR}/tasks/truth.json", "r") as f:
        truth_data = json.load(f)

    loader = UnstructuredExcelLoader(REFERENCE)
    document = loader.load()
    document_chunks = RecursiveCharacterTextSplitter().split_documents(document)
    document_chunks = [doc.page_content for doc in document_chunks]

    # model = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY)
    model = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
    evaluator_llm = LangchainLLMWrapper(model)

    # https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
    coverage_scorer = SimpleCriteriaScore(
        name="coverage_score",
        definition="Score from 0 to 5 based on how well the response captures the vulnerabilities mentioned in the reference.",
        llm=evaluator_llm,
    )
    compliance_scorer = SimpleCriteriaScore(
        name="compliance_score",
        definition="Score from 0 to 5 based on how well the test_case in the response aligns with the retrieved contexts.",
        llm=evaluator_llm,
    )
    metrics = [
        Faithfulness(),
        # FactualCorrectness(atomicity="low", coverage="low"),
        coverage_scorer,
        compliance_scorer,
    ]

    summary_metrics = {}
    for k, (status, result) in results.items():
        logger.info(f"Running AI evaluation on {k}")
        if k not in truth_data:
            logger.warning(f"{k} is not in task/truth.json. Skipping this...")
            continue

        result = filter_json(result)
        truth = truth_data.get(k)
        dataset = [
            {
                "user_input": "",
                "response": json.dumps(result),
                "reference": json.dumps(truth),
                "retrieved_contexts": document_chunks,
            }
        ]
        eval_dataset = EvaluationDataset.from_list(dataset)
        eval_result = evaluate(dataset=eval_dataset, metrics=metrics, llm=evaluator_llm)
        eval_result = (
            eval_result.to_pandas()
            .iloc[:, -len(metrics) :]
            .to_dict(orient="records")[0]
        )
        summary_metrics[k] = eval_result

    return summary_metrics
