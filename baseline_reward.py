import argparse
import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch.multiprocessing as mp
from gen_format_dataset import LLMProblem
from transformers import AutoTokenizer
from utils import gen_batch, get_datetime_str


def gen_grader_prompt(problem: str, solution: str, llm_solution: str) -> str:
    return f"""PROBLEM
{problem}

ANSWER KEY
{solution}

STUDENT SOLUTION
{llm_solution}

INSTRUCTIONS
You are a highly intelligent and astute grader that is never fooled by students. You don't care about showing work; you just care about the right final answer. If the student solution EXACTLY matches the answer key (the value that is boxed in the answer key), output "STUDENT CORRECT". Otherwise, output "STUDENT INCORRECT". Keep your response short.
"""


def compute_reward(grader_output: str, llm_solution: str) -> float:
    if "STUDENT CORRECT" not in grader_output:
        return 0.0

    # n_thought_tokens = llm_solution.find("</thoughts>") - llm_solution.find(
    #     "<thoughts>"
    # )
    # return np.exp(-n_thought_tokens / 2000)
    return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--eval", action="store_true", default=False)
    return parser.parse_args()


@dataclass
class MathSol:
    filename: str
    llm_solution: str
    llm_initial_prompt: str
    grading_prompt: Optional[str] = None
    grader_output: Optional[str] = None
    reward: Optional[float] = None


def main():
    mp.set_start_method("spawn")

    args = parse_args()
    assert os.path.exists(args.model_path)

    if args.eval:
        filename = "MATH/test.json"
    else:
        filename = "MATH/train.json"

    with open(filename, "r") as f:
        data = json.load(f)

    problem_list: list[LLMProblem] = []
    for i in range(len(data)):
        problem = data[i]["data"]

        problem_list.append(
            LLMProblem(
                problem=problem["problem"],
                llm_initial_prompt=problem["problem"],
                solution=problem["solution"],
                filename=data[i]["filename"],
            )
        )

    print("len(problem_list)", len(problem_list))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    N_RUNS = 20 if not args.eval else 3
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
        args.model_path,
        prompt_list,
        max_tokens=2048,
        dtype="float32",
    )

    math_sols: list[MathSol] = []
    math_sols_for_grading: list[MathSol] = []

    for i in range(len(outputs)):
        problem_idx = i // N_RUNS
        llm_solution = outputs[i].outputs[0].text
        sol = MathSol(
            filename=problem_list[problem_idx].filename,
            llm_solution=llm_solution,
            llm_initial_prompt=problem_list[problem_idx].llm_initial_prompt,
        )
        math_sols.append(sol)

        valid_output = (
            llm_solution.count("<thoughts>") == 1
            and llm_solution.count("</thoughts>") == 1
            and llm_solution.find("<thoughts>") < llm_solution.find("</thoughts>")
        )

        if not valid_output:
            sol.reward = 0
        else:
            user_output = llm_solution.split("</thoughts>")[1].strip()
            grading_prompt = gen_grader_prompt(
                problem_list[problem_idx].problem,
                problem_list[problem_idx].solution,
                user_output,
            )
            sol.grading_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": grading_prompt}],
                tokenize=False,
                add_generation_prompt=False,
            )
            math_sols_for_grading.append(sol)

    outputs = gen_batch(
        "meta-llama/Llama-3.1-8B-Instruct",
        [each.grading_prompt for each in math_sols_for_grading],
        max_tokens=500,
        dtype="bfloat16",
    )

    for i in range(len(outputs)):
        val = outputs[i].outputs[0].text
        math_sols_for_grading[i].reward = compute_reward(
            val, math_sols_for_grading[i].llm_solution
        )
        math_sols_for_grading[i].grader_output = val

    problem_to_reward = defaultdict(list)

    for each in math_sols:
        problem_to_reward[each.filename].append(each.reward)

    for k in problem_to_reward:
        print(
            k,
            np.mean(problem_to_reward[k]),
            np.std(problem_to_reward[k]),
            problem_to_reward[k],
        )

    filename = os.path.join(
        "solution_rollouts" if not args.eval else "eval_solution_rollouts",
        f"solution_rollouts_{get_datetime_str()}_{len(math_sols)}.pt",
    )

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print("saved to", filename)
    with open(filename, "wb") as f:
        if args.eval:
            pickle.dump({"math_sols": math_sols, "model_path": args.model_path}, f)
        else:
            pickle.dump(math_sols, f)


if __name__ == "__main__":
    main()

""""
Algo:
1. 30 rollouts per problem (500 problems)
2. compute the advantage of each rollout
3. do like 2-3 epochs over the data using PPO and largish batch sizes
4. repeat



ALGO 2:
1. 20 rollouts per problem
2. take the top 1k problems with the highest std reward (20k samples)
3. compute advantages for each rollout
4. load both the reference policy and the current policy
5. do like 2-3 epochs over the data using PPO and largish batch sizes
"""
