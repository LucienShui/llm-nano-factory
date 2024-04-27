from argparse import ArgumentParser, Namespace
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, BitsAndBytesConfig
from transformers import HfArgumentParser, TrainingArguments, set_seed, Trainer
from dataclasses import dataclass, field
from torch.utils.data import Dataset as TorchDataset
from typing import Dict, List, Any, Optional
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import torch


@dataclass
class Arguments:
    model_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})
    task_type: str = field(metadata={"help": "Task type", "choices": ["sft"]})
    max_seq_length: str = field(metadata={"help": "Max sequence length"})
    train_file: str = field(metadata={"help": "Train file"})
    train_mode: str = field(metadata={"help": "Train mode", "choices": ["full", "lora"]})
    lora_rank: int = field(metadata={"help": "Lora rank"}, default=8)
    lora_target: str = field(metadata={"help": "Lora target"}, default="q_proj,v_proj")
    quantization_bit: Optional[int] = field(metadata={"help": "Quantization bit", "choices": [4, 8]}, default=None)


class Dataset(TorchDataset):
    def __init__(self, json_l_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_seq_length: int = max_seq_length
        with open(json_l_path, encoding="utf-8") as f:
            self.str_data_list = f.readlines()

    def __len__(self):
        return len(self.str_data_list)

    def __getitem__(self, idx: int):
        str_data = self.str_data_list[idx]
        data = json.loads(str_data)
        messages = data["messages"]

        assert len(messages) > 0 and len(messages) % 2 == int(messages[0]['role'] == 'system') \
               and messages[-1]['role'] == 'assistant'

        encoded_ids = []
        target_mask = []

        for i in range(1, len(messages) + 1):
            if messages[i - 1]['role'] == 'system':
                continue
            is_user = messages[i - 1]['role'] == 'user'
            previous_length = len(encoded_ids)
            encoded_ids = self.tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=is_user)

            target_mask += [1 - int(is_user)] * (len(encoded_ids) - previous_length)

        assert len(target_mask) == len(encoded_ids)
        return encoded_ids, target_mask

    def collator(self, batch: List[Dict[str, list]]) -> Dict[str, Any]:
        length_list = list(map(lambda x: min(len(x[0]), self.max_seq_length), batch))
        max_length = max(length_list)

        b_input_ids, b_attn_mask, b_target_mask = [], [], []

        for (input_ids, target_mask), length in zip(batch, length_list):
            padding_length = max_length - length

            b_input_ids.append(input_ids[:length] + [self.tokenizer.pad_token_id] * padding_length)
            b_attn_mask.append([1] * length + [0] * padding_length)
            b_target_mask.append(target_mask[:length] + [0] * padding_length)

        t_b_input_ids, t_b_attn_mask, t_b_target_mask = map(
            lambda x: torch.tensor(x, dtype=torch.long),
            (b_input_ids, b_attn_mask, b_target_mask)
        )
        return {
            "input_ids": t_b_input_ids,
            "attention_mask": t_b_attn_mask,
            "labels": torch.where(t_b_target_mask == 1, t_b_input_ids, -100)
        }


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    return args


def get_trainer(config_file_path: str):
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, train_args = parser.parse_json_file(config_file_path)
    set_seed(train_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        load_in_4bit=args.quantization_bit == 4,
        load_in_8bit=args.quantization_bit == 8
    )

    if args.quantization_bit is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_args.gradient_checkpointing)

    if args.train_mode != "full":
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=args.lora_target.split(","),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.task_type == "pretrain":
        dataset = Dataset(args.train_file, tokenizer, args.max_seq_length)
    else:
        dataset = Dataset(args.train_file, tokenizer, args.max_seq_length)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=dataset.collator,
        tokenizer=tokenizer
    )

    return trainer


def main():
    trainer = get_trainer(get_args().config)
    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == '__main__':
    main()
