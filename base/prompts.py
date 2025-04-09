from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate

OWASP_TEMPLATE = """
    Using the following pieces of retrieved context on OWASP Application 
    Security Verification standards, answer the user's question accordingly.
    """


FEWSHOT_EXAMPLES = [
    {
        "question": "What is the purpose of 'Review Webserver Metafiles for Information Leakage'?",
        "answer": "- Identify hidden or obfuscated paths and functionality through the analysis of metadata files (robots.txt, <META> tag, sitemap.xml) - Extract and map other information that could lead to a better understanding of the systems at hand."
    },
    {
        "question": "What is the purpose of 'Review Webpage Content for Information Leakage'?",
        "answer": "- Review webpage comments, metadata, and redirect bodies to find any information leakage - Gather JavaScript files and review the JS code to better understand the application and to find any information leakage. - Identify if source map files or other front-end debug files exist."
    },
    {
        "question": "What is the purpose of 'Test HTTP Strict Transport Security'?",
        "answer": "- Review the HSTS header and its validity. - Identify HSTS header on Web server through HTTP response header: curl -s -D- https://domain.com/ | grep Strict"
    },
    {
        "question": "What is the purpose of 'Test File Permission'?",
        "answer": "- Review and Identify any rogue file permissions. - Identify configuration file whose permissions are set to world-readable from the installation by default."
    },
    {
        "question": "What is the purpose of 'Test Role Definitions'?",
        "answer": "- Identify and document roles used by the application. - Attempt to switch, change, or access another role. - Review the granularity of the roles and the needs behind the permissions given."    
    },
    {
        "question": "What is the purpose of 'Test User Registration Process'?",
        "answer": "- Verify that the identity requirements for user registration are aligned with business and security requirements. - Validate the registration process."
    },
    {
        "question": "What is the purpose of 'Testing for Weak Lock Out Mechanism'?",
        "answer": "- Evaluate the account lockout mechanism's ability to mitigate brute force password guessing. - Evaluate the unlock mechanism's resistance to unauthorized account unlocking."
    },
    {
        "question": "What is the purpose of 'Testing for Bypassing Authentication Schema'?",
        "answer": "- Ensure that authentication is applied across all services that require it. - Force browsing (/admin/main.php, /page.asp?authenticated=yes), Parameter Modification, Session ID prediction, SQL Injection"
    },
    {
        "question": "What is the purpose of 'Testing for Session Management Schema'?",
        "answer": "- Gather session tokens, for the same user and for different users where possible. - Analyze and ensure that enough randomness exists to stop session forging attacks. - Modify cookies that are not signed and contain information that can be manipulated."
    },
    {
        "question": "What is the purpose of 'Testing for Cross Site Request Forgery'?",
        "answer": "- Determine whether it is possible to initiate requests on a user's behalf that are not initiated by the user. - Conduct URL analysis, Direct access to functions without any token."
    },
    {
        "question": "What is the purpose of 'Testing for Reflected Cross Site Scripting'?",
        "answer": "- Identify variables that are reflected in responses. - Assess the input they accept and the encoding that gets applied on return (if any)."
    },
    {
        "question": "What is the purpose of 'Testing for Stored Cross Site Scripting'?",
        "answer": "- Identify stored input that is reflected on the client-side. - Assess the input they accept and the encoding that gets applied on return (if any)."
    },
	
	
]

few_shot_chat_messsage_prompt = FewShotChatMessagePromptTemplate(
    examples = FEWSHOT_EXAMPLES,
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{question}"),
        ("ai", "{answer}")]
    )
)

OWASP_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", OWASP_TEMPLATE),
        few_shot_chat_messsage_prompt,
        ("human","\Question: {question}\nContext: {context}\nAnswer:"),
    ]
)
