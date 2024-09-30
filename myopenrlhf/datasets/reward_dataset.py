from typing import Callable

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

from .utils import exist_and_not_none, zero_pad_sequences

# from transformers import AutoTokenizer
# from myopenrlhf.utils.utils import make_mental_reward_dataset

def preprocess_data(data):
    # stanfordnlp/SHP
    if exist_and_not_none(data, "human_ref_A"):
        prompt = "Human: " + data["history"] + "\nAssistant: "
        preferA = bool(data["labels"])
        chosen = data["human_ref_A"] if preferA else data["human_ref_B"]
        reject = data["human_ref_B"] if preferA else data["human_ref_A"]
    # HuggingFaceH4/ultrafeedback_binarized
    elif exist_and_not_none(data, "prompt_id"):
        prompt = data["prompt"] if exist_and_not_none(data, "prompt") else ""
        chosen = data["chosen"][1]["content"]
        reject = data["rejected"][1]["content"]
    # Anthropic/hh-rlhf
    # tasksource/oasst1_pairwise_rlhf_reward
    elif exist_and_not_none(data, "chosen") and exist_and_not_none(data, "rejected"):
        prompt = data["prompt"] if exist_and_not_none(data, "prompt") else ""
        if prompt.startswith("prompter:"):
            prompt = prompt.replace("prompter:", "\nHuman: ").replace("assistant:", "\nAssistant: ") + "\nAssistant: "

        chosen = data["chosen"]
        reject = data["rejected"]
    # lvwerra/stack-exchange-paired
    elif exist_and_not_none(data, "response_j"):
        prompt = "Human: " + data["question"] + "\nAssistant: "
        chosen = data["response_j"]
        reject = data["response_k"]
    # lmsys/chatbot_arena_conversations
    elif exist_and_not_none(data, "winner") and exist_and_not_none(data, "conversation_a"):

        def process_chatbot_arena_conversations(lll):
            result = []
            for l in lll:
                result.append(l["role"].replace("user", "Human: ").replace("assistant", "Assistant: "))
                result.append(l["content"])
            return "\n".join(result)

        prompt = ""
        chosen = data["conversation_a"] if data["winner"] == "model_a" else data["conversation_b"]
        reject = data["conversation_b"] if data["winner"] == "model_a" else data["conversation_a"]
        chosen = process_chatbot_arena_conversations(chosen)
        reject = process_chatbot_arena_conversations(reject)
    # openai/webgpt_comparisons
    elif exist_and_not_none(data, "answer_0") and exist_and_not_none(data, "answer_1"):
        prompt = "Human: " + data["question"]["full_text"] + "\nAssistant: "
        chosen = data["answer_0"] if data["score_0"] > data["score_1"] else data["answer_1"]
        reject = data["answer_1"] if data["score_0"] > data["score_1"] else data["answer_0"]
    # damo/CValues-Comparison https://www.modelscope.cn/datasets/damo/CValues-Comparison/quickstart
    elif exist_and_not_none(data, "pos_resp") and exist_and_not_none(data, "neg_resp"):
        prompt = "Human: " + data["prompt"] + "\nAssistant: "
        chosen = data["pos_resp"]
        reject = data["neg_resp"]
    else:
        raise ValueError("reward_dataset key error")

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    return prompt, chosen, reject, margin


def mental_preprocess_data(data):
    prompt = "Human: " + data["query"] + "\nAssistant: "
    response1 = data['response_1']
    response2 = data['response_2']

    cor = data['correctness']
    info = data['informativeness']
    pro = data['professionality']

    cor_margin = data['correctness_margin']
    info_margin = data['informativeness_margin']
    pro_margin = data['professionality_margin']

    return prompt, response1, response2, cor, info, pro, cor_margin, info_margin, pro_margin


class MentalRewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, strategy) -> None:
        super().__init__()
        self.prompts = []
        self.response1 = []
        self.response2 = []
        self.cor = []
        self.info = []
        self.pro = []
        self.cor_margins = []
        self.info_margins = []
        self.pro_margins = []

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        #for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
        for data in tqdm(dataset):
            prompt, response1, response2, cor, info, pro, cor_margin, info_margin, pro_margin = mental_preprocess_data(data)
            self.prompts.append(prompt)
            self.response1.append(response1)
            self.response2.append(response2)
            self.cor.append(cor)
            self.info.append(info)
            self.pro.append(pro)
            self.cor_margins.append(cor_margin)
            self.info_margins.append(info_margin)
            self.pro_margins.append(pro_margin)

    def __len__(self):
        length = len(self.response1)
        return length

    def __getitem__(self, idx):
        prompt, response1, response2, cor, info, pro, cor_margin, info_margin, pro_margin = \
            self.prompts[idx], self.response1[idx], self.response2[idx], self.cor[idx], self.info[idx], self.pro[idx]\
                , self.cor_margins[idx], self.info_margins[idx], self.pro_margins[idx]

        response1 = prompt + str(response1) + " " + self.tokenizer.eos_token
        response1_token = self.tokenizer(
            response1,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        response2 = prompt + str(response2) + " " + self.tokenizer.eos_token
        response2_token = self.tokenizer(
            response2,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        response1_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        response2_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        response1_token["attention_mask"][0][-1] = True
        response2_token["attention_mask"][0][-1] = True
        return (
            response1_token["input_ids"],
            response1_token["attention_mask"],
            response2_token["input_ids"],
            response2_token["attention_mask"],
            cor,
            info,
            pro,
            cor_margin,
            info_margin,
            pro_margin
        )

    def collate_fn(self, item_list):
        response1_ids = []
        response1_masks = []
        response2_ids = []
        response2_masks = []
        cors = []
        infos = []
        pros = []
        cor_margins = []
        info_margins = []
        pro_margins = []

        for response1_id, response1_mask, response2_id, response2_mask, cor, info, pro, cor_margin, info_margin, pro_margin in item_list:
            response1_ids.append(response1_id)
            response2_ids.append(response2_id)
            response1_masks.append(response1_mask)
            response2_masks.append(response2_mask)
            cors.append(cor)
            infos.append(info)
            pros.append(pro)
            cor_margins.append([int(i) for i in cor_margin.split(',')])
            info_margins.append([int(i) for i in info_margin.split(',')])
            pro_margins.append([int(i) for i in pro_margin.split(',')])

        response1_ids = zero_pad_sequences(response1_ids, value=self.tokenizer.pad_token_id)
        response1_masks = zero_pad_sequences(response1_masks)
        response2_ids = zero_pad_sequences(response2_ids, value=self.tokenizer.pad_token_id)
        response2_masks = zero_pad_sequences(response2_masks)
        return response1_ids, response1_masks, response2_ids, response2_masks, torch.LongTensor(cors), torch.LongTensor(infos),\
            torch.LongTensor(pros), torch.tensor(cor_margins, dtype=torch.float32), torch.tensor(info_margins, dtype=torch.float32)\
            , torch.tensor(pro_margins, dtype=torch.float32)


def new_mental_preprocess_data(data):
    prompt1 = "Human: " + data["query"] + "\nAssistant: "
    prompt2 = "\nAspect: " + data['aspect']
    chosen_response = data['chosen_response']
    rejected_response = data['rejected_response']

    margin = data['margins']

    return prompt1, prompt2, chosen_response, rejected_response, margin


class NewMentalRewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, strategy, margin_hyper=1.5) -> None:
        super().__init__()
        self.prompts1 = []
        self.prompts2 = []
        self.chosens = []
        self.rejects = []
        self.margins = []
        self.aspects = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.margin_hyper = margin_hyper

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt1, prompt2, chosen, reject, margin = new_mental_preprocess_data(data)

            if not chosen or not reject:
                continue
            self.prompts1.append(prompt1)
            self.prompts2.append(prompt2)
            self.chosens.append(chosen)
            self.rejects.append(reject)
            self.margins.append(margin)

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt1, prompt2, chosen, reject, margin = self.prompts1[idx],\
            self.prompts2[idx], self.chosens[idx], self.rejects[idx], self.margins[idx]

        chosen = prompt1 + chosen + prompt2 + " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        reject = prompt1 + reject + prompt2 + " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        if 'correctness' in prompt2:
            aspect = 0
        elif 'informativeness' in prompt2:
            aspect = 1
        elif 'professionality' in prompt2:
            aspect = 2
        else:
            print('wrong!dhskskhclsfjslfjls')
        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            margin,
            aspect
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        margins = []
        aspects = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, margin, aspect in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            aspects.append(aspect)
            m1, m2 = margin.split(',')
            margins.append((int(m2)-int(m1))*self.margin_hyper)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks)
        reject_ids = zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, torch.tensor(margins, dtype=torch.float32), torch.tensor(aspects)


def preprocess_DR_data(data):
    prompt1 = 'Human: Consider this post: "' + data["text"] + '" Question: Does the poster suffer from depression?\nAssistant: '
    prompt2 = "\nAspect: "
    response = data['gpt-3.5-turbo']

    return prompt1, prompt2, response


class HumanDRDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int) -> None:
        super().__init__()
        self.prompts1 = []
        self.prompts2 = []
        self.responses = []
        self.aspects = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for data in tqdm(dataset['train']):
            prompt1, prompt2, response = preprocess_DR_data(data)
            for aspect in ['fluency', 'completeness', 'reliability', 'overall']:
                self.prompts1.append(prompt1)
                self.prompts2.append(prompt2)
                self.responses.append(response)
                self.aspects.append(aspect)
    def __len__(self):
        length = len(self.prompts1)
        return length

    def __getitem__(self, idx):
        prompt1, prompt2, response, aspect = self.prompts1[idx],\
            self.prompts2[idx], self.responses[idx], self.aspects[idx]

        chosen = prompt1 + response + prompt2 + aspect + " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        for chosen_id, chosen_mask in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks)
        return chosen_ids, chosen_masks


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, strategy) -> None:
        super().__init__()
        self.prompts = []
        self.chosens = []
        self.rejects = []
        self.margins = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, chosen, reject, margin = preprocess_data(data)
            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)
            self.margins.append(margin)

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, margin = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.margins[idx]

        chosen = prompt + chosen + " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        reject = prompt + reject + " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True
        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            margin,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        margins = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, margin in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            margins.append(margin)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks)
        reject_ids = zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, torch.tensor(margins, dtype=torch.float32)
