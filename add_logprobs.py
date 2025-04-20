import argparse
import os
import pickle

import numpy as np
import torch
from accelerate import Accelerator
from baseline_reward import MathSol
from ppo import RLDatapoint, RLDataset, convert_to_datapoints, get_high_signal_problems
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    assert os.path.exists(args.model_path)
    assert os.path.exists(args.data)

    with open(args.data, "rb") as f:
        math_sols: list[MathSol] = pickle.load(f)

    print("mean reward in dataset", np.mean([each.reward for each in math_sols]))

    math_sols, problem_to_rewards = get_high_signal_problems(math_sols)

    print("len(math_sols)", len(math_sols))
    print("loading dataset")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    math_sols: list[RLDatapoint] = convert_to_datapoints(
        math_sols, tokenizer, problem_to_rewards
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    accelerator = Accelerator(mixed_precision="no")

    GPU_BATCH_SIZE = 28
    dataloader = DataLoader(
        math_sols,
        batch_size=GPU_BATCH_SIZE,
        shuffle=False,
        collate_fn=RLDataset.collate,
    )

    model, dataloader = accelerator.prepare(model, dataloader)
    all_logprobs = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            assert input_ids.shape[1:] == (2048,)

            logits = model(input_ids=input_ids).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            assert log_probs.shape[1:] == (2048, 128257)
            log_probs = torch.gather(
                log_probs, dim=-1, index=input_ids.unsqueeze(-1)
            ).squeeze(-1)
            assert log_probs.shape[1:] == (2048,)

            gathered_logprobs = accelerator.gather(log_probs).cpu()
            gathered_input_ids = accelerator.gather(input_ids).cpu()

            if accelerator.is_main_process:
                print("step", step, "of", len(dataloader))
                all_logprobs.extend(zip(gathered_logprobs, gathered_input_ids))

    if accelerator.is_main_process:
        print(len(math_sols), len(all_logprobs))
        for each, (lp, input_ids) in zip(math_sols, all_logprobs):
            assert lp.shape == (2048,)
            each.logprobs = torch.tensor(lp.numpy()) # otherwise the size blows up??
            assert (each.toks == input_ids).all()

    with open(f"rl_datapoints/{os.path.basename(args.data)}", "wb") as f:
        pickle.dump(math_sols, f)


if __name__ == "__main__":
    main()
