import click
import time
import json

# import testing
import testing_async as testing
from summary import summary
from tasks import template
from analysis.analysis import AnalysisAgent
from enums import EnumEncoder

from loguru import logger
from typing import Dict, Optional


def run_analysis(debug, model, task, randomise):
    agent = AnalysisAgent(model)
    if debug:
        interview_answers_dict = template.generate_template_for_debug(
            task, randomise=randomise
        )
    else:
        interview_answers_dict = template.generate_template(task, randomise=randomise)

    print(interview_answers_dict)

    start_time = time.monotonic()
    code_dict = agent.analyze_multiple_interviews(interview_answers_dict)
    end_time = time.monotonic()

    logger.info(
        f"Took {end_time - start_time} seconds to generate test scripts for: {list(code_dict.keys())}"
    )

    # for key, code in code_dict.items():
    #     formatted_code = json.dumps(code, indent=4)
    #     logger.info(f"Generated code for {key}:\n{formatted_code}\n")

    output_path = "output/test_scripts.json"
    with open(output_path, "w") as json_file:
        json.dump(code_dict, json_file, indent=4)
    logger.info(f"Testing scripts are saved at: {output_path}")

    return code_dict


def run_testing(debug, code_dict: Optional[Dict[str, str]] = None):
    if debug:
        code_dict = template.generate_scripts_for_debug()

    for name, script in code_dict.items():
        print(f"\nScript: {name}\n{'='*40}\n{script[:]}")  #

    start_time = time.monotonic()
    results = testing.run_multiple_tests(code_dict)
    end_time = time.monotonic()

    logger.info(f"Took {end_time - start_time} seconds to run all tests")

    output_path = "output/test_results.json"
    with open(output_path, "w") as json_file:
        json.dump(results, json_file, indent=4, cls=EnumEncoder)
    logger.info(f"Testing results are saved at: {output_path}")

    return results


@click.command()
@click.option("--model", default="base", help="AI model to use.")
@click.option("--task", default="simple", help="Sites to test on.")
@click.option(
    "--randomise",
    is_flag=True,
    default=False,
    help="Whether to include randomised scoping questionnaire responses (default is False).",
)
@click.option(
    "--debug", is_flag=True, default=False, help="Debug mode (default is False)."
)
def main(model, task, randomise, debug):
    if debug:
        logger.warning(
            "Debug mode for development is enabled. Do not use this when running ZAP tests."
        )

    code_dict = run_analysis(debug, model, task, randomise)
    results = run_testing(debug, code_dict=code_dict)

    summary.show_results(results=results)


if __name__ == "__main__":
    main()
