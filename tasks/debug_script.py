import random
import json


def get_working_results():
    output = {
        "type": "result",
        "message": "Simulating a completed test case",
    }
    print(json.dumps(output))


def get_failed_results():
    output = {
        "type": "error",
        "message": "Simulating a failed test case",
    }
    print(json.dumps(output))


def raise_exceptions():
    raise ValueError("Simulating a failed test run")


def main():

    num_working = random.randint(1, 3)
    num_failed = random.randint(1, 3)
    functions = [get_working_results] * num_working + [get_failed_results] * num_failed

    if random.randint(0, 1) == 1:
        random.shuffle(functions)
    else:
        functions = [raise_exceptions]

    for func in functions:
        func()


if __name__ == "__main__":
    main()
