from abc import abstractmethod

from autogen import UserProxyAgent
from pydantic import BaseModel


class BaseReasonerAgent(BaseModel):

    @abstractmethod
    def return_agent(self):
        pass

    def return_user_proxy(self):
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=10,
        )

        return user_proxy
