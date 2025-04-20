import argparse
import collections
import os
import pickle
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tqdm
import utils
from accelerate import Accelerator
from baseline_reward import MathSol
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RLDatapoint:
    filename: str
    toks: torch.tensor
    advantage: float
    num_toks: int
    logprobs: Optional[torch.tensor] = None


class RLDataset(Dataset):
    def __init__(self, data: list[RLDatapoint]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(batch):
        """
        batch: List of RLDatapoint objects
        Returns a dictionary of batched tensors.
        """
        # filenames = [dp.filename for dp in batch]
        # (B, T) shape after stacking
        input_ids = torch.stack([dp.toks for dp in batch], dim=0)
        advantages = torch.tensor([dp.advantage for dp in batch], dtype=torch.float)
        num_toks = torch.tensor([dp.num_toks for dp in batch], dtype=torch.long)
        logprobs = torch.stack([dp.logprobs for dp in batch], dim=0) if batch[0].logprobs is not None else None

        return {
            # "filenames": filenames,
            "input_ids": input_ids,
            "advantages": advantages,
            "num_toks": num_toks,
            "logprobs": logprobs,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    return parser.parse_args()


def train_loop(
    policy: AutoModelForCausalLM,
    # ref_policy: AutoModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    accelerator: Accelerator,
):
    optimizer.zero_grad()

    kl_weight = 0.02
    kl_vals = []
    ratio_list = []
    advantages_list = []
    policy_loss_list = []
    clamp_ratio = []

    # btv = (GPU_BATCH_SIZE, 2048, 128257)

    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(policy):
            input_ids = batch["input_ids"].to(policy.device)
            advantages = batch["advantages"].to(policy.device)
            old_action_log_probs = batch["logprobs"].to(policy.device)
            # assert input_ids.shape == (GPU_BATCH_SIZE, 2048)
            # assert advantages.shape == (GPU_BATCH_SIZE,)
            # assert old_action_log_probs.shape == (GPU_BATCH_SIZE, 2048)

            logits = policy(input_ids=input_ids).logits  # (B, T, V)
            # assert logits.shape == btv, f"{logits.shape} != {btv}"
            log_probs = torch.log_softmax(logits, dim=-1)  # (B, T, V)
            # assert log_probs.shape == btv

            action_log_probs = log_probs.gather(
                dim=-1, index=input_ids.unsqueeze(-1)
            ).squeeze(-1)  # (B, T)

            with torch.no_grad():
                kl = action_log_probs - old_action_log_probs
                # assert kl.shape == btv[:2]

            # assert action_log_probs.shape == btv[:2]
            # assert old_action_log_probs.shape == btv[:2]

            for i in range(len(batch["num_toks"])):
                num_toks = batch["num_toks"][i]
                # assert (input_ids[i, num_toks:] == 128256).all()
                # assert (input_ids[i, :num_toks] != 128256).all()
                kl[i, num_toks:] *= 0.0
                action_log_probs[i, num_toks:] *= 0.0
                old_action_log_probs[i, num_toks:] *= 0.0

            action_log_probs = action_log_probs.sum(dim=-1)
            with torch.no_grad():
                old_action_log_probs = old_action_log_probs.sum(dim=-1)
                kl = kl.sum(dim=-1) / batch["num_toks"].to(policy.device)  # (B,)

            kl_vals.extend(kl.tolist())
            # assert kl.shape == (GPU_BATCH_SIZE,)
            # assert advantages.shape == (GPU_BATCH_SIZE,)
            # assert action_log_probs.shape == (
            #     GPU_BATCH_SIZE,
            # ), f"{action_log_probs.shape} != {(GPU_BATCH_SIZE,)}"
            # assert old_action_log_probs.shape == (
            #     GPU_BATCH_SIZE,
            # ), f"{old_action_log_probs.shape} != {(GPU_BATCH_SIZE,)}"

            advantages = advantages - kl_weight * kl  # (B,)

            # whiten advantages
            all_advantages = accelerator.gather(advantages.detach())
            advantages = (advantages - all_advantages.mean()) * torch.rsqrt(
                all_advantages.var() + 1e-8
            )

            ratios = torch.exp(action_log_probs - old_action_log_probs)  # (B,)
            # assert ratios.shape == (GPU_BATCH_SIZE,)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
            loss = -torch.minimum(surr1, surr2).mean()
            accelerator.backward(loss / 10)

            if accelerator.is_main_process:
                with torch.no_grad():
                    num_ratios = torch.sum((ratios < 0.8) | (ratios > 1.2)).item()
                    clamp_ratio.append(num_ratios / len(ratios))

            ratio_list.extend(ratios.tolist())
            advantages_list.extend(advantages.tolist())
            policy_loss_list.append(loss.item())

            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), max_norm=1000)
            optimizer.step()
            optimizer.zero_grad()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                if accelerator.is_main_process:
                    print("step", step, "of", len(dataloader))
                    print("grad_norm", grad_norm.item())
                    print(
                        "ratios",
                        list(zip(ratio_list, advantages_list)),
                        np.mean([a * b for a, b in zip(ratio_list, advantages_list)]),
                    )
                    print(
                        np.mean(clamp_ratio[-20:]),
                        np.mean(advantages_list),
                        np.std(advantages_list),
                    )
                    print("policy_loss", np.mean(policy_loss_list[-50:]))
                    ratio_list = []
                    advantages_list = []

                    mean_kl = np.mean(kl_vals[-20:])
                    print("mean_kl", mean_kl)
                # if len(kl_vals) > 300:
                #     mean_kl = np.mean(kl_vals[-150:])
                #     if mean_kl > 0.1:
                #         kl_weight *= 1.5
                #         print("updating kl_weight", kl_weight)
                #         kl_vals = []
                #     elif mean_kl < 0.01:
                #         kl_weight /= 1.5
                #         print("updating kl_weight", kl_weight)
                #         kl_vals = []


GPU_BATCH_SIZE = 5
GRAD_ACCUM_STEPS = 20
NUM_EPOCHS = 40


def train_and_save_model(model_path: str, rl_datapoints: list[RLDatapoint]):
    start_train_time = utils.get_datetime_str()
    accelerator = Accelerator(
        mixed_precision="no", gradient_accumulation_steps=GRAD_ACCUM_STEPS
    )

    torch.manual_seed(23945)
    train_dataloader = DataLoader(
        RLDataset(rl_datapoints),
        batch_size=GPU_BATCH_SIZE,
        shuffle=True,
        collate_fn=RLDataset.collate,
    )
    torch.manual_seed(accelerator.process_index + 8394)

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-9)
    model.eval()

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    accelerator.wait_for_everyone()

    for epoch in range(NUM_EPOCHS):
        print(f"epoch {epoch}")
        start_time = time.time()
        train_loop(model, optimizer, train_dataloader, accelerator)
        end_time = time.time()
        if accelerator.is_main_process:
            print("time taken in minutes", (end_time - start_time) / 60)

        train_dataloader = accelerator.free_memory(train_dataloader)

        # reshuffle for next batch
        torch.manual_seed(123490 + epoch)
        train_dataloader = accelerator.prepare(
            DataLoader(
                RLDataset(rl_datapoints),
                batch_size=GPU_BATCH_SIZE,
                collate_fn=RLDataset.collate,
                shuffle=True,
            )
        )
        torch.manual_seed(accelerator.process_index + 12990 + epoch)

        if accelerator.is_main_process:
            folder = f"/data/arjunpatrawala/ppo_model/epoch_{epoch}_{start_train_time}"
            os.makedirs(folder, exist_ok=True)
            print("saving model", folder)
            accelerator.unwrap_model(model).save_pretrained(folder)
            tokenizer.save_pretrained(folder)


def get_high_signal_problems(
    math_sols: list[MathSol],
) -> list[str]:
    problem_to_rewards = collections.defaultdict(list)

    for each in math_sols:
        problem_to_rewards[each.filename].append(each.reward)

    TOP_N_PROBS = 1500

    # get top 1k problems with highest std
    problem_std_rewards = [(k, np.std(v)) for k, v in problem_to_rewards.items()]
    problem_std_rewards.sort(key=lambda item: item[1], reverse=True)
    # print(
    #     [
    #         (v, np.mean(problem_to_rewards[k]), k)
    #         for k, v in problem_std_rewards[:TOP_N_PROBS]
    #     ]
    # )
    problems = set([k for k, _ in problem_std_rewards[:TOP_N_PROBS]])
    print("lowest problem", problem_std_rewards[TOP_N_PROBS])
    assert problem_std_rewards[TOP_N_PROBS][1] < problem_std_rewards[0][1]

    return [each for each in math_sols if each.filename in problems], problem_to_rewards


def convert_to_datapoints(
    math_sols: list[MathSol],
    tokenizer: AutoTokenizer,
    problem_to_rewards: dict[str, list[float]],
) -> list[RLDatapoint]:
    rl_datapoints: list[RLDatapoint] = []

    for each in tqdm.tqdm(math_sols):
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": each.llm_initial_prompt},
                {"role": "assistant", "content": each.llm_solution},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        toks = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=2048,
        )
        assert toks.input_ids.shape == (
            1,
            2048,
        ), f"toks.input_ids.shape: {toks.input_ids.shape}"
        original_length = torch.sum(toks.attention_mask, dim=1)

        normalized_advantage = (
            each.reward - np.mean(problem_to_rewards[each.filename])
        ) / (np.std(problem_to_rewards[each.filename]) + 1e-8)

        rl_datapoints.append(
            RLDatapoint(
                filename=each.filename,
                toks=toks.input_ids[0],
                advantage=normalized_advantage,
                num_toks=original_length[0].item(),
            )
        )

    return rl_datapoints


def main():
    args = parse_args()
    assert os.path.exists(args.model_path)
    assert os.path.exists(args.data)

    with open(args.data, "rb") as f:
        math_sols: list[MathSol] = pickle.load(f)

    # print("mean reward in dataset", np.mean([each.reward for each in math_sols]))

    # math_sols, problem_to_rewards = get_high_signal_problems(math_sols)

    # print("len(math_sols)", len(math_sols))
    # print("loading dataset")

    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # math_sols: list[RLDatapoint] = convert_to_datapoints(
    #     math_sols, tokenizer, problem_to_rewards
    # )

    print(math_sols[0])
    print(
        np.percentile(
            [each.advantage for each in math_sols], [0, 5, 15, 25, 50, 75, 85, 95, 100]
        )
    )

    # print(
    #     np.percentile(
    #         [np.mean(r) for k, r in problem_to_rewards.items()],
    #         [0, 5, 15, 25, 50, 75, 85, 95, 100],
    #     )
    # )

    train_and_save_model(args.model_path, math_sols)


if __name__ == "__main__":
    main()
