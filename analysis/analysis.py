from base.model_handler import AIModelHandler
from base.owasp_model_handler import OWASPModelHandler
from .prompts import ANALYSIS_PROMPT, ZAP_TEMPLATE_SCRIPT
from loguru import logger
import re
import asyncio


class AnalysisAgent:
    def __init__(self, model_handler: str = "base"):
        if model_handler == "base":
            self.model_handler = AIModelHandler()
        elif model_handler == "rag":
            self.model_handler = OWASPModelHandler()
        else:
            raise ValueError(f"Invalid model handler: {model_handler}")

    def analyze_interview(self, interview_answers):

        prompt = ANALYSIS_PROMPT.format(
            template_script=ZAP_TEMPLATE_SCRIPT, interview_answers=interview_answers
        )
        raw_output = self.model_handler.query_model(prompt)
        parsed_output = self._parse_output(raw_output)
        num_test_cases = self._get_num_test_cases(parsed_output)
        logger.info(f"Generated {num_test_cases} test cases")

        return parsed_output

    async def analyze_multiple_interviews_async(self, interview_answers_dict):
        tasks = []
        for interview_answers in interview_answers_dict.values():
            prompt = ANALYSIS_PROMPT.format(
                template_script=ZAP_TEMPLATE_SCRIPT, interview_answers=interview_answers
            )
            tasks.append(
                asyncio.create_task(self.model_handler.query_model_async(prompt))
            )

        results = await asyncio.gather(*tasks)
        results = [self._parse_output(result) for result in results]
        results = dict(zip(interview_answers_dict.keys(), results))
        return results

    def analyze_multiple_interviews(self, interview_answers_dict):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(
            self.analyze_multiple_interviews_async(interview_answers_dict)
        )
        loop.close()
        return results

    def _parse_output(self, raw_output):
        try:
            match = re.search(
                r"```(?:python)?\s*(.*?)\s*```", raw_output, flags=re.DOTALL
            )
            if match:
                code = match.group(1)
            else:
                code = raw_output
            return code.strip()
        except Exception as e:
            logger.warning("Failed to parse raw data from model: {}", e)
            return "Error: Failed to parse model output"

    def _get_num_test_cases(self, code):
        test_functions = re.findall(r"^\s*def\s+(test_\w+)\s*\(", code, re.MULTILINE)
        return len(test_functions)
