from argparse import ArgumentParser, Namespace
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from transformers import HfArgumentParser, TrainingArguments, set_seed, Trainer
from dataclasses import dataclass, field
from torch.utils.data import Dataset as TorchDataset
from typing import Dict, List, Any
import json
import torch


@dataclass
class Arguments:
    model_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})
    task_type: str = field(metadata={"help": "Task type", "choices": ["pretrain", "sft"]})


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

        assert len(messages) > 0
        assert len(messages) % 2 == int(messages[0]['role'] == 'system')
        assert messages[-1]['role'] == 'assistant'

        encoded_ids = []
        target_mask = []

        for i in range(1, len(messages) + 1):
            if messages[i - 1]['role'] == 'system':
                continue
            is_user = messages[i - 1]['role'] == 'user'
            previous_length = len(encoded_ids)
            encoded_ids = self.tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=is_user)

            target_mask += [int(is_user)] * (len(encoded_ids) - previous_length)

        assert len(target_mask) == len(encoded_ids)
        return {'input_ids': encoded_ids, 'target_mask': target_mask}

    def collator(self, batch: List[Dict[str, list]]) -> Dict[str, Any]:
        length_list = list(map(lambda x: len(x["input_ids"]), batch))
        max_length = min(max(length_list), self.max_seq_length)

        b_input_ids, b_target_mask = [], []

        for each in batch:
            input_ids = each["input_ids"][:max_length]
            padding_length = max_length - len(input_ids)

            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            target_mask = each['target_mask'][:max_length] + [0] * padding_length

            b_input_ids.append(input_ids)
            b_target_mask.append(target_mask)

        t_b_input_ids, t_b_target_mask = map(lambda x: torch.tensor(x, dtype=torch.long), (b_input_ids, b_target_mask))
        return {
            "input_ids": t_b_input_ids,
            "target_mask": t_b_target_mask,
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    set_seed(train_args.seed)

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
    trainer.train()
    trainer.save_model()
    trainer.save_metrics()
    trainer.save_state()


if __name__ == '__main__':
    main()
