import plotly.express as px
from typing import Dict
from enums import TestStatus
from loguru import logger

OUTPUT_PATH = "output/test_chart.png"


def show_completion(results: Dict):
    # TODO: include charts to show summary results

    completed = [
        result for result in results.values() if result[0] == TestStatus.COMPLETED
    ]
    prop_completed = len(completed) / len(results)
    print(
        f"Proportion of completed scripts: {len(completed)}/{len(results)} ({prop_completed * 100} %)\n"
    )

    total_working = 0
    total_available = 0
    for key, (status, json_data_list) in results.items():
        if status == TestStatus.COMPLETED:
            working = [
                json_data
                for json_data in json_data_list
                if json_data.get("type") == "result"
            ]
            prop_working = len(working) / len(json_data_list)
            print(
                f"Testing for {key} completed {len(working)}/{len(json_data_list)} ({round(prop_working * 100)}%) of test cases generated."
            )
            total_working += len(working)
            total_available += len(json_data_list)
        else:
            print(f"Testing for {key} failed.")

    if len(completed) != 0:
        print(
            f"Overall working test cases: {total_working}/{total_available} ({(total_working / total_available) * 100} %)\n"
        )


def show_coverage(results: Dict):
    raise NotImplementedError
