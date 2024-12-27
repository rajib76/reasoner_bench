import json

from benchmark.mcts_reasoner_benchmark import MCTSReasonerBenchmark

if __name__=="__main__":
    with open("data/simple_bench_public.json", "r") as f:
        benchmark_data = json.load(f)

    eval_data = benchmark_data["eval_data"]
    instructions = """
    Give the final answer in the below mentioned JSON FORMAT.

    {"answer" : "X,where X is one of the letters A,B,C,D,E, or F"}

    REMEMBER TO SHARE ONLY THE OUTPUT IN JSON FORMAT. DO NOT ADD ANYTHING ELSE. DO NOT ADD THE EXPLANATION OF YOUR RESPONSE.
    """

    rb = MCTSReasonerBenchmark()
    rb.run_benchmark(eval_list=eval_data, instructions=instructions)
