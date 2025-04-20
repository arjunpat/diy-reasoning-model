import argparse
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, device_map="cuda")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda")

    while True:
        try:
            question = input("Enter your question: ")
            if question == "exit":
                break
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                return_tensors="pt",
                tokenize=True,
            )

            output = model.generate(
                input_ids.to(model.device), max_length=4096, do_sample=True
            )
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
            print(answer)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
