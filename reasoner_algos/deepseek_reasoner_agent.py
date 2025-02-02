import requests
from reasoner_algos.base import BaseReasonerAgent


class DeepseekClient:
    def __init__(self, url="http://localhost:11434/api/generate", model="deepseek-r1:8b"):
        """
        Initialize the client with the API endpoint and model name.
        """
        self.url = url
        self.model = model
        self.headers = {"content-type": "application/json"}

    def generate(self, prompt, stream=False):
        """
        Send a prompt to the Deepseek API and return the response.

        Args:
            prompt (str): The prompt to be sent to the model.
            stream (bool): Whether to use streaming mode (default is False).

        Returns:
            dict or str: The API response, either as parsed JSON or text.
        """
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }

        try:
            response = requests.post(self.url, headers=self.headers, json=data)
            response.raise_for_status()  # Raise an error for bad status codes

            # Assuming the response is in JSON format:
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return None


class DeepSeekReasonerAgent(BaseReasonerAgent):

    def return_agent(self):
        deepseek_agent = DeepseekClient()
        return deepseek_agent


# Example usage:
if __name__ == '__main__':
    # Create an instance of the client
    deepseek_agent = DeepSeekReasonerAgent()
    agent = deepseek_agent.return_agent()

    # prompt = "9.11 and 9.8, which is greater?"
    prompt="""
    Beth places four whole ice cubes in a frying pan at the start of the first minute, then five at the start of the second minute and some more at the start of the third minute, but none in the fourth minute. If the average number of ice cubes per minute placed in the pan while it was frying a crispy egg was five, how many whole ice cubes can be found in the pan at the end of the third minute?
A. 30
B. 0
C. 20
D. 10
E. 11
F. 5
    """



    result = agent.generate(prompt=prompt)
    print("Response from Deepseek API:")
    print(result["response"])
