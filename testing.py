import tempfile
import subprocess
import os
import json
from enums import TestStatus
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_output(line, queue):
    cleaned_line = line.decode().strip()
    try:
        json_data = json.loads(cleaned_line)
        queue.append(json_data)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON output: {cleaned_line}")


def read_output(stream, queue):
    while True:
        line = stream.readline()
        if not line:
            break
        process_output(line, queue)


def run_tests_parallel(code, key):
    tmp_file_path = None
    queue = []

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(code)
            tmp_file_path = tmp_file.name

        logger.info(f"Temporary test file created at: {tmp_file_path}")

        process = subprocess.Popen(
            ["python", "-u", tmp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        read_output(process.stdout, queue)
        read_output(process.stderr, queue)

        process.wait()
        exit_code = process.returncode

        if exit_code == 0:
            logger.info(f"Script executed successfully")
            return {key: (TestStatus.COMPLETED, queue)}
        else:
            logger.error(f"Script failed with exit code {exit_code}")
            return {key: (TestStatus.FAILED, queue)}
    except Exception as e:
        logger.error(f"Failed to run tests: {e}")
        return {key: (TestStatus.FAILED, [])}

    finally:
        try:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
                logger.info("Temporary test file removed")
        except Exception as e:
            logger.error(f"Failed to remove temporary file: {e}")


def run_multiple_tests(code_dict):
    test_results = dict()

    with ThreadPoolExecutor(max_workers=4) as executor:

        futures = [
            executor.submit(run_tests_parallel, code, key)
            for key, code in code_dict.items()
        ]

        for future in as_completed(futures):
            result = future.result()
            result_key = list(result.keys())[0]
            logger.info(f"Completed test for {result_key}")
            test_results.update(result)

    return test_results
