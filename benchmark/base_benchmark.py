import json
from abc import abstractmethod

from pydantic import BaseModel


class BaseBenchmark(BaseModel):

    @abstractmethod
    def run_benchmark(self,eval_list,instructions=None,out_file_name="benchmark_result.jsonl"):
        pass

    def _write_benchmark_output(self,json_line,out_file_name):
        with open(out_file_name, "a") as f:
            f.write(json.dumps(json_line) + "\n")


