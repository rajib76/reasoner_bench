from dotenv import load_dotenv
from openai import OpenAI

from reasoner_algos.base import BaseReasonerAgent

load_dotenv()


class O1ReasonerAgent(BaseReasonerAgent):
    config_list: list = []

    def return_agent(self):
        api_key = self.config_list[0]["api_key"]
        o1_agent = OpenAI(
            api_key=api_key
        )

        return o1_agent

