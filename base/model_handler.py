from openai import OpenAI, AsyncOpenAI
import os
import requests
from loguru import logger


class AIModelHandler:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_TOKEN")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=self.api_key
        )
        self.model = "deepseek/deepseek-r1-distill-llama-70b:free"
        logger.info("Initialised base model handler")

    def get_usage_info(self):
        url = "https://openrouter.ai/api/v1/auth/key"
        response = requests.get(
            url, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()

    def query_model(self, prompt):
        completion = self.client.chat.completions.create(
            extra_body={},
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = completion.choices[0].message.content
        return content

    async def query_model_async(self, prompt):
        completion = await self.async_client.chat.completions.create(
            extra_body={},
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = completion.choices[0].message.content
        return content
