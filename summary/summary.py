import sys
import json
from pathlib import Path
from typing import Dict
from loguru import logger
from collections import Counter
from enums import TestStatus
from summary.eval_model import run_eval_with_ai

BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.append(str(BASE_DIR))


def show_completion(results: Dict):
    # TODO: include charts to show summary results

    output = ""
    completed = [
        result for result in results.values() if result[0] == TestStatus.COMPLETED
    ]
    prop_completed = len(completed) / len(results)
    output += f"Proportion of completed scripts: {len(completed)}/{len(results)} ({prop_completed * 100} %)\n\n"

    total_working = 0
    total_available = 0
    for key, (status, json_data_list) in results.items():
        if status == TestStatus.COMPLETED:
            working = [
                json_data
                for json_data in json_data_list
                if json_data.get("type") == "result"
            ]

            total_working += len(working)
            total_available += len(json_data_list)

            if len(json_data_list) == 0:
                output += f"Testing for {key} has 0 test cases generated.\n"
            else:
                prop_working = len(working) / len(json_data_list)
                output += f"Testing for {key} completed {len(working)}/{len(json_data_list)} ({round(prop_working * 100)}%) of test cases generated.\n"
        else:
            output += f"Testing for {key} failed.\n"

    if len(completed) != 0:
        output += f"Overall working test cases: {total_working}/{total_available} ({(total_working / total_available) * 100} %)\n"

    return output


def show_coverage(results: Dict):

    summary_metrics = run_eval_with_ai(results)
    with open(f"{BASE_DIR}/output/test_summary_metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=4)

    output = "Notable metrics\n"
    output += "\t- faithfulness: Measures how factually consistent the testing results are with the OWASP mappings. It ranges from 0 to 1.\n"
    output += "\t- factual_correctness: Measures the factual accuracy of the testing results with the website vulnerabilities.\n"
    output += "\t- coverage_score: Score from 0 to 5 based on how well the testing captures the vulnerabilities embedded in the websites.\n"
    output += "\t- compliance_score: Score from 0 to 5 based on how well the test cases in the testing aligns with the OWASP mappings.\n"
    output += "\n"

    total = Counter()
    for k, v in summary_metrics.items():
        output += f"Testing for {k} = {v}\n"
        total += Counter(v)
    average = {k: v / len(results) for k, v in total.items()}
    output += f"Average = {average}\n"

    return output


def show_results(results: Dict, completion: bool = True, coverage: bool = True):
    output = "\nSumary Results\n---------------------------------\n"
    if completion:
        output += show_completion(results)
    output += "\n"
    if coverage:
        output += show_coverage(results)

    print(output)
