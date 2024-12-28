import os

from autogen import ReasoningAgent
from dotenv import load_dotenv

from reasoner_algos.base import BaseReasonerAgent

load_dotenv()


class BEAMReasonerAgent(BaseReasonerAgent):
    config_list: list = []

    def return_agent(self):
        beam_agent = ReasoningAgent(
            name="reason_agent",
            verbose=False,
            llm_config={"config_list": self.config_list},
            reason_config={"method": "dfs", "max_depth": 3},  # Using DFS
        )
        return beam_agent
