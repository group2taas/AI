# AI

A simple CLI tool for refine AI and run penetration testing

## Structure

```
.
├── analysis              <- code to generate scoping responses to test cases
│   ├── prompts.py        <- the place to improve on prompt engineering
│   ...
├── base                  <- the part where most improvements should take place
├── source                <- modify this if you need new data source for RAG
├── tasks                 <- contain sites to run pentest on (WIP)
├── testing.py            <- script for pentration testing
├── evaluate.py           <- main script for the CLI
...

```

## Setting Up

1. Clone the repo
   `git clone <repository_url>`
   `cd AI`

2. Set up Virtual Env
   \
    `python3 -m venv venv` (linux/macOS)
   `source venv/bin/activate`
   \
    `python -m venv venv` (windows)
   `venv\Scripts\activate`
3. Install
   `pip install -r requirements.txt`

## Usage (WIP)

```
Usage: evaluate.py [OPTIONS]

Options:
  --model TEXT  AI model to use.
  --task TEXT   Sites to test on.
  --randomise   Whether to include randomised scoping questionnaire responses
                (default is False).
  --debug       Debug mode (default is True).
  --help        Show this message and exit.

```
