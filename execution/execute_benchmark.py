import asyncio
import json
import os
from typing import List
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from benchmark.base_benchmark import BaseBenchmark
from benchmark.beam_reasoner_benchmark import BEAMReasonerBenchmark
from benchmark.lats_reasoner_benchmark import LATSReasonerBenchmark
from benchmark.mcts_reasoner_benchmark import MCTSReasonerBenchmark
from benchmark.o1_reasoner_benchmark import O1ReasonerBenchmark

load_dotenv()


class BenchMarkExecution(BaseModel):

    def summarize_results(self, agent_list: List[BaseBenchmark]):
        observation_count = []
        matches = []
        agent_names = []
        accuracy = []
        for agent in agent_list:
            file_name = agent.return_benchmark_name() + ".jsonl"
            total_observations = 0
            total_matches = 0
            with open(file_name, "r") as f:
                for line in f:
                    total_observations = total_observations + 1
                    line_json = json.loads(line)
                    ground_truth = line_json["ground_truth"]
                    predicted_truth = line_json["predicted_truth"]
                    if ground_truth == predicted_truth:
                        total_matches = total_matches + 1

            observation_count.append(total_observations)
            matches.append(total_matches)
            agent_names.append(file_name)
            accuracy.append(total_matches / total_observations * 100)

        data = {'Name': agent_names, 'observation': observation_count, "matches": matches, "accuracy": accuracy}
        df = pd.DataFrame(data)

        return df

    async def run_execution(self,
                            eval_data, agent_list: List[BaseBenchmark],
                            instructions=None,
                            config_list=None):
        for agent in agent_list:
            out_file = agent.return_benchmark_name() + ".jsonl"
            await agent.run_benchmark(eval_list=eval_data,
                                      instructions=instructions,
                                      out_file_name=out_file,
                                      config_list=config_list)

        df = self.summarize_results(agent_list)

        return df


if __name__ == "__main__":
    eval_file = "../data/simple_bench_public.json"
    agent_list = [MCTSReasonerBenchmark(), LATSReasonerBenchmark(), BEAMReasonerBenchmark(), O1ReasonerBenchmark()]
    # agent_list = [BEAMReasonerBenchmark()]
    # instructions = """
    # Give the final answer in the below mentioned JSON FORMAT.
    #
    # {"answer" : "X,where X is one of the letters A,B,C,D,E, or F"}
    #
    # REMEMBER TO SHARE ONLY THE OUTPUT IN JSON FORMAT. DO NOT ADD ANYTHING ELSE. DO NOT ADD THE EXPLANATION OF YOUR RESPONSE.
    # """
    # instructions = """
    # Give the final answer in the below mentioned JSON FORMAT.
    #
    # {"answer" : "X"}
    #
    # REMEMBER TO SHARE ONLY THE OUTPUT IN JSON FORMAT. DO NOT ADD ANYTHING ELSE. DO NOT ADD THE EXPLANATION OF YOUR RESPONSE.
    # """

    # The config list only used for the AG2 reasoining agent. for o1 reasoner,
    # o1-preview is harcoded in o1_reasoner_agent.py
    config_list = [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}]

    with open(eval_file, "r") as f:
        benchmark_data = json.load(f)

    eval_data = benchmark_data["eval_data"]
    be = BenchMarkExecution()
    df = asyncio.run(be.run_execution(eval_data=eval_data,
                                      agent_list=agent_list,
                                      instructions=None,
                                      config_list=config_list))

    print(df.head())
