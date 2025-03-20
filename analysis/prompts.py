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
    template="""You are an expert in cybersecurity penetration testing.
Based solely on the interview answers provided and the OWASP guidelines, write a complete pentration testing script in Python that implements multiple comprehensive test cases using Selenium and ZAP API.
Your output must be in the following format strictly. Replace <target_url> with the given application url and add relevant test functions. 
Ensure that each newly added test function includes proper try-except blocks, storing the output or error messages in self.results using the test name as the key.
Ensure that each test function:
- Uses `try-except` blocks for error handling.
- Stores error messages in `self.results` using the test name as the key.
- Properly formats f-strings.
The output should have no explanations, no comments, no extra text.

{template_script}

Interview Answers:
{interview_answers}
"""
)
