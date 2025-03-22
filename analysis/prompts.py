import os
from langchain.prompts import PromptTemplate

def py_to_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    return f"```python\n{code}\n```"

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
ZAP_TEMPLATE_SCRIPT = py_to_markdown(os.path.join(BASE_DIR, "template.py"))

ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["template_script", "interview_answers"],
    template="""
You are an expert in cybersecurity penetration testing.
Based solely on the interview answers provided and the OWASP guidelines, write a complete penetration testing script in Python that implements multiple comprehensive test cases using Selenium and ZAP API.
Your output must strictly follow the provided format. Replace <target_url> with the given application URL and add relevant test functions.

Ensure that each test function:
- Uses `try-except` blocks for robust error handling.
- Stores error messages in `self.results` using the test name as the key.
- Properly formats f-strings and ensures all function calls use correct syntax.
- Ensures all string inputs, including SQL injections, are properly enclosed in quotes.

To avoid timeout errors, invalid JSON responses, and ZAP execution failures, the script must include:
1. Configurable dynamic timeout with exponential backoff retries.
2. ZAP scan status logging and fallback if scans fail.
3. Graceful continuation of test suite even if setup or scans fail.
4. Structured error messages in JSON for all skipped or failed tests.
5. All test outcomes written to `self.results`, even for skipped tests.
6. All variable and function names must be valid Python identifiers (e.g., use `snake_case`, no spaces, no invalid characters).

The output should contain no explanations, no comments, and no extra text.

{template_script}

Interview Answers:
{interview_answers}
"""
)
