import ast
import asyncio
import json
import os
from typing import ClassVar

from pydantic import ConfigDict

from benchmark.base_benchmark import BaseBenchmark
from reasoner_algos.deepseek_reasoner_agent import DeepSeekReasonerAgent
from reasoner_algos.o1_reasoner_agent import O1ReasonerAgent


class DeepSeekReasonerBenchmark(BaseBenchmark):
    deepseek_agent_class: ClassVar = DeepSeekReasonerAgent()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def create_benchmark_output(self, summary_args):
        benchmarking_result = {}
        question_id = summary_args["question_id"]
        prompt = summary_args["prompt"]
        ground_truth = summary_args["ground_truth"]
        out_file = summary_args["out_file"]
        response = summary_args["response"]
        cost = summary_args["cost"]
        answer_option, answer_explanation = self.create_structured_output(response)
        # if "json" in response:
        #     cleaned_string = response.replace("```json", "").replace("```", "").strip()
        #     # Convert the cleaned string to a valid dictionary
        #     data = eval(cleaned_string)  # Use eval cautiously or replace with a safer parser if needed
        #     cleansed_output = data.get('answer')
        #     print("cleansed ", cleansed_output)
        #     benchmarking_result["predicted_truth"] = cleansed_output
        # else:
        #     model_answer = response
        #     response_json = ast.literal_eval(json.loads(json.dumps(model_answer)))
        #     benchmarking_result["predicted_truth"] = response_json["answer"]

        benchmarking_result["predicted_truth"] = answer_option
        benchmarking_result["explanantion"] = answer_explanation
        benchmarking_result["question_id"] = question_id
        benchmarking_result["prompt"] = prompt
        benchmarking_result["ground_truth"] = ground_truth
        benchmarking_result["cost"] = cost

        self._write_benchmark_output(benchmarking_result, out_file_name=out_file)

    async def run_benchmark(self, eval_list,
                            instructions=None,
                            out_file_name="benchmark_result.jsonl",
                            config_list=None):
        for eval in eval_list:
            question_id = eval["question_id"]
            prompt = eval["prompt"]
            ground_truth = eval["answer"]
            print("working on question : ", question_id)
            deepseek_agent = self.deepseek_agent_class.return_agent()
            completion = deepseek_agent.generate(prompt=prompt)
            # input_tokens = completion.usage.prompt_tokens
            # output_tokens = completion.usage.completion_tokens
            # cached_tokens = completion.usage.prompt_tokens_details.cached_tokens
            # # Not a good practice. harcoded values. TODO - MAKE THIS CONFIGURABLE LATER
            # # $15.00 / 1M input tokens
            # # $60.00 / 1M output** tokens
            # # I think I need to take care of the cached tokens also
            # input_token_cost = input_tokens*15*100/1000000
            # output_token_cost = output_tokens * 60*100/1000000
            # cached_token_cost = cached_tokens * 7.50*100/1000000
            # total_cost = input_token_cost + output_token_cost + cached_token_cost
            # print("completion " , completion.usage.total_tokens)

            response = completion["response"]
            print(response)
            summary_args = {"question_id": question_id, "prompt": prompt, "ground_truth": ground_truth,
                            "response": response, "out_file": out_file_name, "cost": 0}

            self.create_benchmark_output(summary_args)

            # return response

    def return_benchmark_name(self):
        return "deepseek_reasoner"


if __name__ == "__main__":
    eval_file = "../data/simple_bench_public.json"
    # config_list = [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}]
    with open(eval_file, "r") as f:
        benchmark_data = json.load(f)
    eval_data = benchmark_data["eval_data"]
    agent = DeepSeekReasonerBenchmark()
    asyncio.run(agent.run_benchmark(eval_list=eval_data,
                                    instructions=None,
                                    out_file_name="deepseek.jsonl",
                                    config_list=None))
