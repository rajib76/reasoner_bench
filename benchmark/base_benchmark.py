import json
import os
from abc import abstractmethod

from pydantic import BaseModel


class BaseBenchmark(BaseModel):
    config_list: list = [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}]

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
        with open(out_file_name, "a") as f:
            f.write(json.dumps(json_line) + "\n")
