import random
import os
import pickle

import numpy as np
from baseline_reward import MathSol

evals = os.listdir("eval_solution_rollouts")

for eval in evals:
    with open(os.path.join("eval_solution_rollouts", eval), "rb") as f:
        data: list[MathSol] = pickle.load(f)

    model = data['model_path']
    math_sols = data['math_sols']

    avg_reward = np.mean([each.reward for each in math_sols])
    std_reward = np.std([each.reward for each in math_sols])

    print(model, avg_reward, std_reward)
    # random.shuffle(math_sols)
    # correct = [each for each in math_sols if each.reward == 1]
    # print(correct[0].llm_initial_prompt)
    # print(correct[0].llm_solution)
    # print(correct[0].grading_prompt)
    # print(correct[0].grader_output)
