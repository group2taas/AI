from langchain_core.prompts import ChatPromptTemplate

OWASP_TEMPLATE = """
    Using the following pieces of retrieved context on OWASP Application 
    Security Verification standards, answer the user's question accordingly.
    """
OWASP_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", OWASP_TEMPLATE),
        (
            "human",
            "\Question: {question}\nContext: {context}\nAnswer:",
        ),
    ]
)
