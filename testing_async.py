import tempfile
import asyncio
import os
import json
from enums import TestStatus
from loguru import logger


async def process_output(line, queue):
    cleaned_line = line.decode().strip()
    try:
        json_data = json.loads(cleaned_line)
        await queue.put(json_data)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON output: {cleaned_line}")


async def read_output(stream, queue):
    while True:
        line = await stream.readline()
        if not line:
            break
        await process_output(line, queue)


async def run_tests_async(code):
    tmp_file_path = None
    queue = asyncio.Queue()  # TODO: check if this is truly thread-safe

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(code)
            tmp_file_path = tmp_file.name

        logger.info(f"Temporary test file created at: {tmp_file_path}")

        async def monitor_process_health(process, timeout=900):
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout)
                return process.returncode
            except asyncio.TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                return -1

        async def run_subprocess():
            sub_process = await asyncio.create_subprocess_exec(
                "python",
                "-u",
                tmp_file_path,
                limit=2**20,  # 1 MiB
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            tasks = [
                asyncio.create_task(read_output(sub_process.stdout, queue)),
                asyncio.create_task(read_output(sub_process.stderr, queue)),
                asyncio.create_task(monitor_process_health(sub_process)),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task exception: {result}")

            return results[-1]

        exit_code = await run_subprocess()
        json_data_list = []
        while not queue.empty():
            json_data_list.append(await queue.get())

        if exit_code == 0:
            logger.info(f"Script executed successfully")
            return TestStatus.COMPLETED, json_data_list
        else:
            logger.error(f"Script failed with exit code {exit_code}")
            return TestStatus.FAILED, json_data_list

    except Exception as e:
        logger.error(f"Failed to run tests: {e}")
        return TestStatus.FAILED, []

    finally:
        try:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
                logger.info("Temporary test file removed")
        except Exception as e:
            logger.error(f"Failed to remove temporary file: {e}")


async def run_multiple_tests_async(code_dict):
    tasks = [run_tests_async(code) for code in code_dict.values()]
    results = await asyncio.gather(*tasks)
    results = dict(zip(code_dict.keys(), results))
    return results


def run_multiple_tests(code):
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    # results = loop.run_until_complete(run_multiple_tests_async(code))
    # loop.close()
    results = asyncio.run(run_multiple_tests_async(code))
    return results
