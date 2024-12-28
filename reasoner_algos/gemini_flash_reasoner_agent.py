import os

from dotenv import load_dotenv
from google import genai

from reasoner_algos.base import BaseReasonerAgent

load_dotenv()


class GeminiFlashReasonerAgent(BaseReasonerAgent):
    config_list: list = []

    def return_agent(self):
        flash_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'), http_options={'api_version':'v1alpha'})

        return flash_client

