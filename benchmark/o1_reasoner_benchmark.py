import ast
import json
from typing import ClassVar

from pydantic import ConfigDict

from benchmark.base_benchmark import BaseBenchmark
from reasoner_algos.o1_reasoner_agent import O1ReasonerAgent


class O1ReasonerBenchmark(BaseBenchmark):
    o1_agent_class: ClassVar = O1ReasonerAgent()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def create_benchmark_output(self, summary_args):
        benchmarking_result = {}
        question_id = summary_args["question_id"]
        prompt = summary_args["prompt"]
        ground_truth = summary_args["ground_truth"]
        out_file = summary_args["out_file"]
        response = summary_args["response"]
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
            if instructions is not None:
                prompt = prompt + "\n" + instructions

            # if config_list is None:
            #     config_list: list = [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}]

            self.o1_agent_class.config_list = config_list
            o1_agent = self.o1_agent_class.return_agent()
            completion = o1_agent.chat.completions.create(
                model="o1-preview",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response = completion.choices[0].message.content
            print(response)
            summary_args = {"question_id": question_id, "prompt": prompt, "ground_truth": ground_truth,
                            "response": response, "out_file": out_file_name}

            self.create_benchmark_output(summary_args)


            # return response

    def return_benchmark_name(self):
        return "o1_reasoner"