import json

if __name__ == "__main__":
    total_matches = 0
    total_observations = 0
    with open("benchmark_result.jsonl", "r") as f:
        for line in f:
            total_observations = total_observations + 1
            line_json = json.loads(line)
            ground_truth = line_json["ground_truth"]
            predicted_truth = line_json["predicted_truth"]

            if (ground_truth == predicted_truth):
                total_matches =  total_matches + 1

    print("total matches: ", total_matches)
    print("total obs: ", total_observations)
    print("% correct: ", total_matches/total_observations*100 , "%")
