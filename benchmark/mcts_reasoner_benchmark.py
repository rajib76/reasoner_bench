import ast
import json
from typing import ClassVar

from pydantic import ConfigDict

from benchmark.base_benchmark import BaseBenchmark
from reasoner_algos.mcts_reasoner_agent import MCTSReasonerAgent


class MCTSReasonerBenchmark(BaseBenchmark):
    mcts_agent_class: ClassVar = MCTSReasonerAgent()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def last_meaningful_msg(self, sender, recipient, summary_args):
        import warnings

        benchmarking_result = {}
        question_id = summary_args["question_id"]
        prompt = summary_args["prompt"]
        ground_truth = summary_args["ground_truth"]
        out_file = summary_args["out_file"]

        if sender == recipient:
            return "TERMINATE"

        summary = ""
        chat_messages = recipient.chat_messages[sender]

        for msg in reversed(chat_messages):
            try:
                content = msg["content"]
                if isinstance(content, str):
                    summary = content.replace("TERMINATE", "")
                elif isinstance(content, list):
                    # Remove the `TERMINATE` word in the content list.
                    summary = "\n".join(
                        x["text"].replace("TERMINATE", "") for x in content if isinstance(x, dict) and "text" in x
                    )
                if summary.strip().rstrip():
                    print("summary ...")
                    print(summary)
                    print("summary ...")
                    benchmarking_result["question_id"] = question_id
                    benchmarking_result["prompt"] = prompt
                    benchmarking_result["ground_truth"] = ground_truth
                    summary_json = ast.literal_eval(json.loads(json.dumps(summary)))
                    benchmarking_result["predicted_truth"] = summary_json["answer"]
                    self._write_benchmark_output(benchmarking_result,out_file_name=out_file)
                    benchmarking_result = {}
                    return summary
            except (IndexError, AttributeError) as e:
                warnings.warn(f"Cannot extract summary using last_msg: {e}. Using an empty str as summary.",
                              UserWarning)

    def run_benchmark(self, eval_list, instructions=None,out_file="benchmark_result.jsonl"):
        for eval in eval_list:
            question_id = eval["question_id"]
            prompt = eval["prompt"]
            ground_truth = eval["answer"]
            if instructions is not None:
                prompt = prompt + "\n" + instructions
            mcts_agent = self.mcts_agent_class.return_agent()
            user_proxy = self.mcts_agent_class.return_user_proxy()
            summary_args = {"question_id": question_id, "prompt": prompt, "ground_truth": ground_truth,
                            "out_file": out_file}
            user_proxy.initiate_chat(mcts_agent, message=prompt, summary_method=self.last_meaningful_msg,
                                     summary_args=summary_args)
