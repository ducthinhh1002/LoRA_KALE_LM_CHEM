import argparse
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model


def load_and_prepare_datasets():
    ds1_dict = load_dataset("batterydata/battery-device-data-qa")
    if "train" in ds1_dict:
        ds1 = ds1_dict["train"]
    else:
        # some versions of the dataset only provide a validation split
        ds1 = ds1_dict[list(ds1_dict.keys())[0]]

    ds2 = load_dataset("avankumar/Battery_NER_70", split="train")
    ds3 = load_dataset("batterydata/paper-abstracts", split="train")

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
    return concatenate_datasets([ds1, ds2, ds3])


def main(args):
    dataset = load_and_prepare_datasets()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length)

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype="auto", device_map="auto"
        )
    else:
        # avoid meta tensor errors when loading on CPU-only setups
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")

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
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
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
    args = parser.parse_args()
    main(args)
