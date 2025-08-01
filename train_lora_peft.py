import argparse
import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

try:
    import wandb  # optional dependency
except ImportError:  # pragma: no cover - wandb is optional
    wandb = None


class PrintLossCallback(TrainerCallback):
    """A simple callback that prints training loss to stdout."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            print(f"Step {state.global_step}: loss = {logs['loss']:.4f}")


def load_and_prepare_datasets():
    ds1_dict = load_dataset("batterydata/battery-device-data-qa")
    if "train" in ds1_dict:
        ds1 = ds1_dict["train"]
    else:
        # some versions of the dataset only provide a validation split
        ds1 = ds1_dict[list(ds1_dict.keys())[0]]

    ds2 = load_dataset("avankumar/Battery_NER_70", split="train")
    ds3 = load_dataset("batterydata/paper-abstracts", split="train")
    ds4 = load_dataset("iamthomaspruyn/battery-electrolyte-qa", split="train")

    def normalize(example):
        if "question" in example and "answer" in example:
            text = example["question"] + "\n" + example["answer"]
        elif "text" in example:
            text = example["text"]
        elif "abstract" in example:
            text = example.get("title", "") + "\n" + example["abstract"]
        else:
            text = "".join(str(v) for v in example.values())
        return {"text": text}

    ds1 = ds1.map(normalize)
    ds2 = ds2.map(normalize)
    ds3 = ds3.map(normalize)
    ds4 = ds4.map(normalize)
    return concatenate_datasets([ds1, ds2, ds3, ds4])


def main(args):
    dataset = load_and_prepare_datasets()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length)

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    if args.wandb_project and wandb is not None:
        wandb.init(project=args.wandb_project, name=args.run_name)
    elif args.wandb_project and wandb is None:
        print("wandb is not installed. Proceeding without logging.")

    # Use a single loading path for both CPU and GPU setups. Loading with
    # ``device_map="auto"`` can create ``meta`` tensors which ``Trainer``
    # later tries to move with ``model.to(device)`` leading to the
    # ``Cannot copy out of meta tensor`` error.  Instead, load the model on
    # the CPU first and let ``Trainer`` handle moving it to the correct
    # device.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype="auto"
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    use_fp16 = torch.cuda.is_available()
    if not use_fp16:
        print("GPU not found. Disabling fp16 training")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        fp16=use_fp16,
        logging_steps=args.logging_steps,
        save_steps=200,
        save_total_limit=2,
        report_to="wandb" if args.wandb_project else None,
        run_name=args.run_name if args.wandb_project else None,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[PrintLossCallback()],
    )

    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="USTC-KnowledgeComputingLab/Llama3-KALE-LM-Chem-8B",
    )
    parser.add_argument("--output_dir", default="./lora_kale_lm_chem")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Steps between logging callbacks")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Project name for Weights & Biases logging")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run name for wandb")
    args = parser.parse_args()
    main(args)
