import os

from autogen import ReasoningAgent
from dotenv import load_dotenv

from reasoner_algos.base import BaseReasonerAgent

load_dotenv()


class MCTSReasonerAgent(BaseReasonerAgent):
    config_list: list = []

    def return_agent(self):
        mcts_agent = ReasoningAgent(
            name="mcts_agent",
            llm_config={"config_list": self.config_list},
            verbose=False,
            reason_config={"method": "mcts", "nsim": 5, "max_depth": 4},
        )

        return mcts_agent
