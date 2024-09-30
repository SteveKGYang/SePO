from typing import Callable

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(tokenizer, data, pretrain_mode=False):
    # Dahoas/full-hh-rlhf
    # iamketan25/open-assistant-instructions
    if exist_and_not_none(data, "prompt") and exist_and_not_none(data, "chosen"):
        prompt = data["prompt"]
        target = data["chosen"]
    # pvduy/sharegpt_alpaca_oa_vicuna_format
    elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "label"):
        prompt = data["prompt"].replace("USER:", "\nHuman: ").replace("ASSISTANT:", "\nAssistant: ")
        target = data["label"].replace("</s>", "")
    # stingning/ultrachat
    elif exist_and_not_none(data, "id") and exist_and_not_none(data, "data"):
        messages = []
        dialogue = data['data']
        roles = ["user", "assistant"]
        for i, sen in enumerate(dialogue[:-1]):
            messages.append({"role": roles[int(i%2)], "content": sen})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        target = dialogue[-1]
    # BelleGroup/train_0.5M_CN
    # LLMs/Alpaca-ShareGPT
    # yahma/alpaca-cleaned
    # QingyiSi/Alpaca-CoT
    elif exist_and_not_none(data, "instruction") and exist_and_not_none(data, "output"):
        input = " " + data["input"] if exist_and_not_none(data, "input") else ""
        prompt = "Human: " + data["instruction"] + input + "\nAssistant: "
        target = data["output"]
    # Open-Orca/OpenOrca
    elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
        prompt = "Human: " + data["system_prompt"] + "\n" + data["question"] + "\nAssistant: "
        target = data["response"]
    # crumb/gpt4all-clean
    # nomic-ai/gpt4all-j-prompt-generations
    elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "response"):
        prompt = "Human: " + data["prompt"] + "\nAssistant: "
        target = data["response"]
    # FreedomIntelligence/phoenix-sft-data-v1
    elif exist_and_not_none(data, "conversations"):
        prompt = ""
        target = ""
        for item in data["conversations"]:
            if item["from"] == "human":
                prompt = "Human: " + item["value"] + "\nAssistant: "
            elif item["from"] == "gpt":
                target = item["value"]
    # EleutherAI/pile
    elif exist_and_not_none(data, "text") and exist_and_not_none(data, "meta"):
        prompt = ""
        target = data["text"]
        pretrain_mode = False  # ignore prompt.replace(xxx)
    # JSON files for decision transformer
    elif exist_and_not_none(data, "input") and exist_and_not_none(data, "output"):
        prompt = data["input"]
        target = data["output"]
    else:
        raise ValueError("sft_dataset key error")

    if pretrain_mode:
        prompt.replace("Human:", " ").replace("\nAssistant:", " ")
    return prompt, target


'''class RRHFMentalDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, strategy, margin_hyper=1.5) -> None:
        super().__init__()
        self.query = []
        self.chosens = []
        self.rejects = []
        self.margins = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.margin_hyper = margin_hyper

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            query, chosen, reject, margin = data["query"], data['chosen_response'], data['rejected_response'], data['margins']
            self.query.append(query)
            self.chosens.append(chosen)
            self.rejects.append(reject)
            self.margins.append(margin)

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        query, chosen, reject, margin = self.query[idx],\
            self.chosens[idx], self.rejects[idx], self.margins[idx]

        query_token = self.tokenizer(
            query,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        chosen = chosen + " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        reject = str(reject) + " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        # chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        # reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        # chosen_token["attention_mask"][0][-1] = True
        # reject_token["attention_mask"][0][-1] = True

        return (
            query_token["input_ids"],
            query_token["attention_mask"],
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            margin
        )

    def collate_fn(self, item_list):
        query_ids = []
        query_masks = []
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        margins = []
        for query_id, query_mask, chosen_id, chosen_mask, reject_id, rejects_mask, margin in item_list:
            query_ids.append(query_id)
            query_masks.append(query_mask)
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            m1, m2 = margin.split(',')
            margins.append((int(m2)-int(m1))*self.margin_hyper)

        query_ids = zero_pad_sequences(query_ids, value=self.tokenizer.pad_token_id)
        query_masks = zero_pad_sequences(query_masks)
        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks)
        reject_ids = zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks)
        return query_ids, query_masks, chosen_ids, chosen_masks, reject_ids, \
            rejects_masks, torch.tensor(margins, dtype=torch.float32)'''


class RRHFMentalDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        target_aspect,
        pretrain_mode=False,
    ) -> None:
        super().__init__()
        self.prompts = []
        self.chosens = []
        self.rejects = []
        self.margins = []
        self.prompt_ids_lens = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, chosen, reject, margin, aspect = \
                data["query"], data['chosen_response'], data['rejected_response'], data['margins'], data['aspect']

            if not self.pretrain_mode:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].sum().item()
            else:
                prompt_ids_len = 0

            if not self.pretrain_mode:
                if aspect.strip() != target_aspect:
                    continue
                # filter the sample whose length is greater than max_length (2 for answer length)
                if prompt_ids_len >= self.max_length - 2:
                    continue
                if not prompt or not reject:
                    continue

            self.prompt_ids_lens.append(prompt_ids_len)
            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        chosen = self.chosens[idx]
        reject = self.rejects[idx]

        chosen_input_token = self.tokenizer(
            prompt + " " +chosen + " " + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        reject_input_token = self.tokenizer(
            prompt + " " + str(reject) + " " + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        info = {"input": prompt, "chosen": chosen, 'reject': reject}
        # to avoid EOS_token truncation
        chosen_input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_input_token["attention_mask"][0][-1] = True
        return prompt_ids_len, chosen_input_token["input_ids"], chosen_input_token["attention_mask"],\
            reject_input_token["input_ids"], reject_input_token["attention_mask"], info

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        chosen_input_ids = []
        reject_input_ids = []
        chosen_attention_masks = []
        reject_attention_masks = []
        infos = {"input": [], "chosen": [], 'reject': []}

        for prompt_ids_len, chosen_input_id, chosen_attention_mask, reject_input_id, reject_attention_mask, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            chosen_input_ids.append(chosen_input_id)
            chosen_attention_masks.append(chosen_attention_mask)
            reject_input_ids.append(reject_input_id)
            reject_attention_masks.append(reject_attention_mask)
            infos["input"].append(info["input"])
            infos["chosen"].append(info["chosen"])
            infos["reject"].append(info["reject"])

        chosen_input_ids = zero_pad_sequences(chosen_input_ids, "right", self.tokenizer.pad_token_id)
        reject_input_ids = zero_pad_sequences(reject_input_ids, "right", self.tokenizer.pad_token_id)
        chosen_attention_masks = zero_pad_sequences(chosen_attention_masks, "right")
        reject_attention_masks = zero_pad_sequences(reject_attention_masks, "right")
        return prompt_ids_lens, chosen_input_ids, chosen_attention_masks, reject_input_ids, reject_attention_masks, infos


class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        pretrain_mode=False,
    ) -> None:
        super().__init__()
        self.prompts = []
        self.targets = []
        self.prompt_ids_lens = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, target = preprocess_data(tokenizer, data, pretrain_mode)

            if not self.pretrain_mode:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].sum().item()
            else:
                prompt_ids_len = 0

            if not self.pretrain_mode:
                # filter the sample whose length is greater than max_length (2 for answer length)
                if prompt_ids_len >= self.max_length - 2:
                    continue
                if not prompt or not target:
                    continue

            self.prompt_ids_lens.append(prompt_ids_len)
            self.prompts.append(prompt)
            self.targets.append(target)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        target = self.targets[idx]

        input_token = self.tokenizer(
            prompt + target + " " + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        info = {"input": prompt, "output": target}
        # to avoid EOS_token truncation
        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True
        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id, attention_mask, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])

        input_ids = zero_pad_sequences(input_ids, "left", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "left")
        return prompt_ids_lens, input_ids, attention_masks, infos
