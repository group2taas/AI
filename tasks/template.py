import random
from typing import Dict
import os


# TODO: complete this questionaire
scoping_questions = {
    "Environment for Testing": ["Production", "Staging", "Development"],
    "Application Type": ["Web", "Mobile", "API"],
    "Authentication method": ["Username/Password", "SSO", "OAuth", "Certificate-based"],
    "Session Management": ["Cookies", "JWT"],
    "Session Timeout (mins)": [0, 24 * 60],
    "Total number of input fields": [0, 100],
    "Number of endpoints": [0,100],
    "Authentication Required": [True, False],
    "Rate Limiting": [True, False],
    "Documentation Available": [True, False],
    "Hosting": ["Cloud", "On-premises", "Hybrid"]
}


def generate_single_interview_answer(url: str, randomise: bool = False) -> str:
    interview_answers = f"Question: Production URL\nAnswer: {url}\n\n"
    if randomise:
        for key, value in scoping_questions.items():
            if isinstance(value[0], int):
                choice = random.randint(value[0], value[1])
            else:
                choice = random.choice(value)
            interview_answers += f"Question: {key}\nAnswer: {choice}\n\n"
    return interview_answers


def generate_template(task: str, randomise: bool = False) -> Dict[str, str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = f"{base_dir}/{task}.txt"
    urls = {}
    with open(filepath, "r") as file:
        for line in file:
            if line.strip(): 
                key, value = line.strip().split(maxsplit=1)
                urls[key] = value
            else:
                break

    return {
        key: generate_single_interview_answer(url, randomise)
        for key, url in urls.items()
    }


def generate_template_for_debug(task: str, randomise: bool = False) -> Dict[str, str]:
    return {
        "example1": generate_single_interview_answer("https://example.com", randomise),
        # "example2": generate_single_interview_answer("https://example.com", randomise),
        # "example3": generate_single_interview_answer("https://example.com", randomise),
        # "example4": generate_single_interview_answer("https://example.com", randomise),
    }


def generate_scripts_for_debug(num_scripts=4) -> Dict[str, str]:

    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{base_dir}/debug_script.py", "r") as file:
        content = file.read()
    
    return {f"example{i}": content for i in range(1, num_scripts + 1)}
