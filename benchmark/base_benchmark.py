import ast
import json
import os
from abc import abstractmethod
from typing import ClassVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()


class AnswerOption(BaseModel):
    answer_option: str = Field(description="The selected answer option which can be on of A,B,C,D,E,F")
    answer_explanation: str = Field(description="The explnantion of the answer")


class BaseBenchmark(BaseModel):
    config_list: list = [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}]
    client: ClassVar = OpenAI()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def run_benchmark(self, eval_list,
                      instructions=None,
                      out_file_name="benchmark_result.jsonl",
                      config_list=None):
        pass

    @abstractmethod
    def return_benchmark_name(self):
        pass

    def _write_benchmark_output(self, json_line, out_file_name):
        print("file name ", out_file_name)
        with open(out_file_name, "a") as f:
            f.write(json.dumps(json_line) + "\n")

    def _update_benchmark_output(self,cost,out_file_name):
        print("file name ", out_file_name)
        with open(out_file_name, "r") as file:
            lines = file.readlines()

        updated_lines = []
        for line in lines:
            data = json.loads(line)
            data["cost"] = cost
            updated_lines.append(json.dumps(data) + '\n')

        with open(out_file_name, 'w') as file:
            file.writelines(updated_lines)

            # print("line ")
            # line = file.read()
            # print("line ", line)
            # line_json = ast.literal_eval(json.loads(json.dumps(line)))
            # line_json["cost"] = cost
            # file.write(json.dumps(line_json) + "\n")


    def create_structured_output(self, answer_content):
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "Extract the answer option and the answer explanation from the provided "
                                              "answer_content."},
                {"role": "user", "content": answer_content},
            ],
            response_format=AnswerOption,
        )

        event = completion.choices[0].message.parsed
        print("my event ", event)
        answer_option = event.answer_option
        answer_explanation = event.answer_explanation

        return answer_option,answer_explanation
