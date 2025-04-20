import json
import pickle
import random
from dataclasses import dataclass, field
from datetime import datetime

import huggingface_hub
import torch.multiprocessing as mp
from utils import gen_batch
from vllm import LLM

from api_keys import HF_ACCESS_TOKEN

huggingface_hub.login(token=HF_ACCESS_TOKEN)


COT_PROMPT = "Solve the above math problem using a step-by-step chain of thought approach. Put all of your thinking into ONE scratchpad that starts with <thoughts> and ends with </thoughts>. What you put in the <thoughts> tag will not be shown to the user, so feel free to explore different ideas and techniques. If you use 5000 thought tokens and cannot arrive upon an answer, tell the user that you are unable to solve the problem. When you are done thinking, end the thoughts tag and explain your solution to the user in a helpful way, walking them through the steps."


def gen_prompt(prompt: str) -> str:
    return f"""{prompt}

{COT_PROMPT}"""


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


@dataclass
class LLMProblem:
    filename: str
    problem: str
    solution: str
    llm_initial_prompt: str
    llm_solution: list[str] = field(default_factory=list)


def main():
    # want to train the model to output in the format desired
    mp.set_start_method("spawn")
    with open("MATH/train.json", "r") as f:
        data = json.load(f)

    problem_list: list[LLMProblem] = []
    for i in range(len(data)):
        problem = data[i]["data"]

        prompt = problem["problem"]
        problem_list.append(
            LLMProblem(
                problem=prompt,
                llm_initial_prompt=gen_prompt(prompt),
                solution=problem["solution"],
                filename=data[i]["filename"],
            )
        )

    print("len(problem_list)", len(problem_list))
    random.shuffle(problem_list)

    model = LLM(MODEL_NAME, dtype="bfloat16")
    tokenizer = model.get_tokenizer()

    N_RUNS = 8
    prompt_list = []
    for each in problem_list:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": each.llm_initial_prompt}],
            tokenize=False,
            add_generation_prompt=False,
        )
        for i in range(N_RUNS):
            prompt_list.append(prompt)

    outputs = gen_batch(
        MODEL_NAME,
        model,
        prompt_list,
        max_tokens=2800,
        dtype="bfloat16",
    )

    for i in range(len(outputs)):
        corresponding_problem = i // N_RUNS
        problem_list[corresponding_problem].llm_solution.append(outputs[i])

    dataset = []
    # create fine-tuning dataset
    for each in problem_list:
        for sol in each.llm_solution:
            sol_split = sol.split("<|start_header_id|>assistant<|end_header_id|>")
            if len(sol_split) != 2:
                print(sol_split)
                continue

            response = sol_split[1]

            num_begin_thoughts = response.count("<thoughts>")
            num_end_thoughts = response.count("</thoughts>")
            if num_begin_thoughts != 1 or num_end_thoughts != 1:
                print("num_begin_thoughts", num_begin_thoughts)
                print("num_end_thoughts", num_end_thoughts)
                continue

            if response.find("<thoughts>") > response.find("</thoughts>"):
                print("response.find('<thoughts>') > response.find('</thoughts>')")
                continue

            # want it to start thinking immediately
            updated_response = response[response.find("<thoughts>") :]

            datapoint = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": each.problem},
                    {"role": "assistant", "content": updated_response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            print("-" * 100)
            print(datapoint)
            dataset.append(datapoint)

    print("len(dataset)", len(dataset))
    with open(
        f"format_dataset/train_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}_{len(dataset)}.pkl",
        "wb",
    ) as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
