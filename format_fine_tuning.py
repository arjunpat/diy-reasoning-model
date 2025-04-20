import argparse
import os
import pickle
import random

from gen_format_dataset import COT_PROMPT
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=4096):
        """
        Args:
            texts (list of str): The dataset containing raw strings.
            tokenizer: The Hugging Face tokenizer to process the texts.
            max_length (int): Maximum length of tokenized input.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Tokenizes the text at index `idx` and returns it in PyTorch tensor format.
        """
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Ensure tensors are not empty
        for key, val in encoding.items():
            assert val.shape[1] > 0, f"Empty tensor for key: {key}, idx: {idx}"

        return {key: val.squeeze(0) for key, val in encoding.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset (Pickle file)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save model and logs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name or path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.dataset = [x for x in os.listdir('./format_dataset') if x.endswith('.pkl')][0]
    args.dataset = os.path.join('./format_dataset', args.dataset)

    with open(args.dataset, "rb") as f:
        dataset: list[str] = pickle.load(f)
    print(dataset[0])

    assert isinstance(dataset, list) and all(
        isinstance(s, str) for s in dataset
    ), "Dataset must be a list of strings."

    print("len(dataset):", len(dataset))
    dataset = ["".join(s.split(COT_PROMPT)) for s in dataset if len(s) < 2800]
    print("len(dataset):", len(dataset))

    random.shuffle(dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    train_test_split_n = int(len(dataset) * 0.95)
    train_texts, test_texts = dataset[:train_test_split_n], dataset[train_test_split_n:]

    train_dataset = CustomDataset(train_texts, tokenizer, max_length=2800)
    test_dataset = CustomDataset(test_texts, tokenizer, max_length=2800)

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # Define training arguments
    # first run: 18 batch size; 5e-6 lr; 2-4 grad norm; 6 epochs; .97-.69 eval loss over training
    # second run: 72 batch size; 1e-5 lr;
    # third run: 48 batch size; 5e-5 lr; -> did not work
    # fourth run: 48 batch size; 1e-5 lr;
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=16,
        weight_decay=0.01,
        save_steps=500,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        warmup_steps=200,
        bf16=True,
        push_to_hub=False,
        ddp_find_unused_parameters=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Save final model and tokenizer
    model_path = f"{args.output_dir}/final_model"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    print(f"Model and tokenizer saved to {model_path}")


if __name__ == "__main__":
    main()
