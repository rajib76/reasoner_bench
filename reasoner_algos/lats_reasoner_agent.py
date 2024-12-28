import os

from autogen import ReasoningAgent
from dotenv import load_dotenv

from reasoner_algos.base import BaseReasonerAgent

load_dotenv()


class LATSReasonerAgent(BaseReasonerAgent):
    config_list: list = []

    def return_agent(self):
        lats_agent = ReasoningAgent(
            name="lats_agent",
            llm_config={"config_list": self.config_list},
            verbose=False,
            reason_config={"method": "lats", "nsim": 5, "max_depth": 4},
        )

        return lats_agent
