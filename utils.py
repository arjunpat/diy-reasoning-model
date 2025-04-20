import os
from datetime import datetime

import torch
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams


def get_datetime_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")


def gen_process(
    model: str,
    prompts: list[str],
    max_tokens: int,
    results_queue: mp.Queue,
    rank: int,
    dtype: str,
) -> list[str]:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"
    outputs = LLM(model, dtype=dtype).generate(
        prompts,
        SamplingParams(max_tokens=max_tokens),
        use_tqdm=True,
    )

    # results = [each.outputs[0].text for each in outputs]
    results_queue.put((rank, outputs))


def gen_batch(
    model_path: str,
    prompts: list[str],
    max_tokens: int | None = None,
    dtype: str = "bfloat16",
) -> list[str]:
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs found."

    n_per_gpu = len(prompts) // num_gpus
    assert n_per_gpu > 0

    processes = []
    results = mp.Queue(num_gpus)
    for rank in range(num_gpus):
        start = rank * n_per_gpu
        end = (rank + 1) * n_per_gpu
        if rank == num_gpus - 1:
            end = len(prompts)

        print(f"Starting {rank} of {num_gpus - 1}")
        p = mp.Process(
            target=gen_process,
            args=(model_path, prompts[start:end], max_tokens, results, rank, dtype),
        )
        p.start()
        processes.append(p)

    print("Waiting for processes to finish")
    outputs = []
    for _ in range(len(processes)):
        outputs.append(results.get())

    for p in processes:
        p.join()

    outputs.sort(key=lambda x: x[0])
    # flatten
    outputs = [x for xs in map(lambda x: x[1], outputs) for x in xs]

    return outputs
