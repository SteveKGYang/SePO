import argparse
import os.path

import torch

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_dpo_ref_model(dpo_model_path, ref_model_path):
    dpo_model = AutoModelForCausalLM.from_pretrained(dpo_model_path)
    tokenizer = AutoTokenizer.from_pretrained(dpo_model_path)

    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path)

    return dpo_model, ref_model, tokenizer


def preprocess_data(dataset_name, tokenizer):
    o_prompts = []
    prompts = []
    chosen_responses = []
    rejected_responses = []
    chosen_targets = []
    rejected_targets = []

    if dataset_name == 'HuggingFaceH4/ultrafeedback_binarized':
        dataset = load_dataset(dataset_name, split='train_prefs')
        i = 0
        for data in tqdm(dataset):
            # if i>60:
            #     break
            messages = [
                {"role": "user", "content": data['prompt']}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            context = tokenizer([prompt], return_tensors="pt")
            if len(context.input_ids[0]) > 1900:
                continue
            chosen_response = data['chosen'][1]['content']
            c_response = tokenizer([chosen_response], return_tensors="pt")
            if len(c_response.input_ids[0]) < 8:
                continue
            rejected_response = data['rejected'][1]['content']
            r_response = tokenizer([rejected_response], return_tensors="pt")
            if len(r_response.input_ids[0]) < 8:
                continue
            o_prompts.append(data['prompt'])
            chosen_responses.append(chosen_response+tokenizer.eos_token)
            rejected_responses.append(rejected_response+tokenizer.eos_token)
            prompts.append(prompt)
            chosen_targets.append(prompt+chosen_response+tokenizer.eos_token)
            rejected_targets.append(prompt+rejected_response+tokenizer.eos_token)
            i+=1

    return o_prompts, chosen_responses, rejected_responses, prompts, chosen_targets, rejected_targets


def get_topK(advantage, N_token, k_perc, ids):
    assert advantage.shape[-1] == ids.shape[-1]
    selected_args = {}
    unselected_args = {}
    selected_ids = {}
    unselected_ids = {}

    for k in k_perc:
        selected_args[k] = []
        unselected_args[k] = []
        selected_ids[k] = []
        unselected_ids[k] = []

    selected_advantage = advantage.clone()
    unselected_advantage = advantage.clone()

    flags = [0 for _ in k_perc]
    while sum(flags) < len(flags):
        max_arg = torch.argmax(selected_advantage, dim=-1).detach().cpu()
        for i, k in enumerate(k_perc):
            if len(selected_args[k]) < int(float(k) * N_token):
                selected_args[k].append(int(max_arg))
                selected_ids[k].append(int(ids[max_arg]))
            else:
                flags[i] = 1
        selected_advantage[max_arg] = -100000

    flags = [0 for _ in k_perc]
    while sum(flags) < len(flags):
        min_arg = torch.argmin(unselected_advantage, dim=-1).detach().cpu()
        for i, k in enumerate(k_perc):
            if len(unselected_args[k]) < int(float(k) * N_token):
                unselected_args[k].append(int(min_arg))
                unselected_ids[k].append(int(ids[min_arg]))
            else:
                flags[i] = 1
        unselected_advantage[min_arg] = 100000
    # selected_args.sort()
    # unselected_args.sort()
    for k in k_perc:
        assert len(selected_args[k]) == len(selected_ids[k])
        assert len(unselected_args[k]) == len(unselected_ids[k])

    return selected_args, unselected_args, selected_ids, unselected_ids


def main(args):
    dpo_model_path = args['dpo_model_path']
    ref_model_path = args['ref_model_path']
    device = args['device']
    dataset_name = args['dataset']
    k_perc = args['k_perc'].split(",")
    save_file_name = args['save_dir']

    dpo_model, ref_model, tokenizer = get_dpo_ref_model(dpo_model_path, ref_model_path)
    dpo_model.to(device)
    ref_model.to(device)

    o_prompts, chosen_responses, rejected_responses, prompts, chosen_targets, rejected_targets\
        = preprocess_data(dataset_name, tokenizer)

    chosen_selected_args = {}
    chosen_unselected_args = {}
    rejected_selected_args = {}
    rejected_unselected_args = {}

    chosen_selected_ids = {}
    chosen_unselected_ids = {}
    rejected_selected_ids = {}
    rejected_unselected_ids = {}

    for k in k_perc:
        chosen_selected_args[k] = []
        chosen_unselected_args[k] = []
        rejected_selected_args[k] = []
        rejected_unselected_args[k] = []

        chosen_selected_ids[k] = []
        chosen_unselected_ids[k] = []
        rejected_selected_ids[k] = []
        rejected_unselected_ids[k] = []

    chosen_responses_ids = []
    rejected_responses_ids = []

    for i in range(len(prompts)):
        print(i)
        cu_prompts = prompts[i]
        cu_chosen_targets = chosen_targets[i]
        cu_rejected_targets = rejected_targets[i]

        context = tokenizer([cu_prompts], return_tensors="pt", padding=False)
        context_ids = context.input_ids[0]

        chosen_embed = tokenizer([cu_chosen_targets], return_tensors="pt", padding=False)
        chosen_ids = chosen_embed.input_ids
        if chosen_ids.shape[-1] > 2048:
            chosen_ids = chosen_ids[:, :2048].to(device)
            chosen_ids[:, -1] = 2
        else:
            chosen_ids = chosen_ids.to(device)

        # Chosen advantage calculation.
        dpo_logits = dpo_model(chosen_ids)["logits"].detach()
        ref_logits = ref_model(chosen_ids)["logits"].detach()

        chosen_labels = chosen_ids[:, 1:].cpu()
        dpo_logits = dpo_logits[:, :-1, :].cpu()
        ref_logits = ref_logits[:, :-1, :].cpu()

        dpo_per_token_logps = torch.gather(dpo_logits.log_softmax(-1), dim=2, index=chosen_labels.unsqueeze(2)).squeeze(2)
        ref_per_token_logps = torch.gather(ref_logits.log_softmax(-1), dim=2, index=chosen_labels.unsqueeze(2)).squeeze(2)

        chosen_advantages = dpo_per_token_logps - ref_per_token_logps

        # Filter out prompt logits.
        # prompt_masks = torch.cat([context_masks, torch.zeros(context_masks.shape[0],
        #                chosen_masks.shape[1]-context_masks.shape[1])], dim=-1)
        # prompt_masks = torch.log(1-prompt_masks)
        chosen_advantages = chosen_advantages[0, len(context_ids):]
        chosen_response_id = chosen_ids[0, len(context_ids):].cpu()
        # chosen_selected_advantages = chosen_advantages + prompt_masks
        # chosen_unselected_advantages = chosen_advantages - prompt_masks

        #Select the top-K tokens.
        N_token = len(chosen_advantages)

        selected_args, unselected_args, selected_ids, unselected_ids \
            = get_topK(chosen_advantages, N_token, k_perc,
                       chosen_ids[0, len(context_ids):chosen_ids.shape[-1] - 1].cpu())
        for k in k_perc:
            chosen_selected_args[k].append(selected_args[k])
            chosen_unselected_args[k].append(unselected_args[k])
            chosen_selected_ids[k].append(selected_ids[k])
            chosen_unselected_ids[k].append(unselected_ids[k])
        chosen_responses_ids.append(chosen_response_id)

        k = k_perc[-1]
        for j in range(len(selected_ids)):
            assert chosen_response_id[int(selected_args[k][j])] == selected_ids[k][j]
        for j in range(len(unselected_ids)):
            assert chosen_response_id[int(unselected_args[k][j])] == unselected_ids[k][j]

        rejected_embed = tokenizer([cu_rejected_targets], return_tensors="pt", padding=False)
        # rejected_ids = rejected_embed.input_ids[:, :2048].to(device)
        rejected_ids = rejected_embed.input_ids
        if rejected_ids.shape[-1] > 2048:
            rejected_ids = rejected_ids[:, :2048].to(device)
            rejected_ids[:, -1] = 2
        else:
            rejected_ids = rejected_ids.to(device)

        # Rejected advantage calculation.
        dpo_logits = dpo_model(rejected_ids)["logits"].detach()
        ref_logits = ref_model(rejected_ids)["logits"].detach()

        rejected_labels = rejected_ids[:, 1:].cpu()
        dpo_logits = dpo_logits[:, :-1, :].cpu()
        ref_logits = ref_logits[:, :-1, :].cpu()

        dpo_per_token_logps = torch.gather(dpo_logits.log_softmax(-1), dim=2, index=rejected_labels.unsqueeze(2)).squeeze(
            2)
        ref_per_token_logps = torch.gather(ref_logits.log_softmax(-1), dim=2, index=rejected_labels.unsqueeze(2)).squeeze(
            2)

        rejected_advantages = dpo_per_token_logps - ref_per_token_logps

        # Mask out prompt logits.
        # prompt_masks = torch.cat([context_masks, torch.zeros(context_masks.shape[0],
        #                             rejected_masks.shape[1] - context_masks.shape[1])], dim=-1)
        # prompt_masks = torch.log(1 - prompt_masks)
        # rejected_selected_advantages = rejected_advantages + prompt_masks
        # rejected_unselected_advantages = rejected_advantages - prompt_masks
        rejected_advantages = rejected_advantages[0, len(context_ids):]
        rejected_response_id = rejected_ids[0, len(context_ids):].cpu()

        # Select the top-K tokens.
        N_token = len(rejected_advantages)
        # selected_args, unselected_args, selected_ids, unselected_ids \
        #     = get_topK(rejected_advantages, N_token, k_perc, rejected_ids[0, len(context_ids):rejected_ids.shape[-1]-1].cpu())
        # rejected_selected_args.append(selected_args)
        # rejected_unselected_args.append(unselected_args)
        # rejected_selected_ids.append(selected_ids)
        # rejected_unselected_ids.append(unselected_ids)
        # rejected_responses_ids.append(rejected_response_id)

        selected_args, unselected_args, selected_ids, unselected_ids \
            = get_topK(rejected_advantages, N_token, k_perc,
                       rejected_ids[0, len(context_ids):rejected_ids.shape[-1] - 1].cpu())
        for k in k_perc:
            rejected_selected_args[k].append(selected_args[k])
            rejected_unselected_args[k].append(unselected_args[k])
            rejected_selected_ids[k].append(selected_ids[k])
            rejected_unselected_ids[k].append(unselected_ids[k])
        rejected_responses_ids.append(rejected_response_id)

        k = k_perc[-1]
        for j in range(len(selected_ids)):
            assert rejected_response_id[int(selected_args[k][j])] == selected_ids[k][j]
        for j in range(len(unselected_ids)):
            assert rejected_response_id[int(unselected_args[k][j])] == unselected_ids[k][j]
    # assert len(chosen_selected_args) == len(prompts)
    # assert len(rejected_selected_args) == len(prompts)

    chosen_selected_args_string = {}
    rejected_selected_args_string = {}
    chosen_unselected_args_string = {}
    rejected_unselected_args_string = {}

    chosen_selected_ids_string = {}
    rejected_selected_ids_string = {}
    chosen_unselected_ids_string = {}
    rejected_unselected_ids_string = {}

    for k in k_perc:
        chosen_selected_args_string[k] = []
        chosen_unselected_args_string[k] = []
        rejected_selected_args_string[k] = []
        rejected_unselected_args_string[k] = []

        chosen_selected_ids_string[k] = []
        chosen_unselected_ids_string[k] = []
        rejected_selected_ids_string[k] = []
        rejected_unselected_ids_string[k] = []

    chosen_responses_ids_string = []
    rejected_responses_ids_string = []

    for k in k_perc:
        for chosen in chosen_selected_args[k]:
            chosen_selected_args_string[k].append(",".join([str(i) for i in chosen]))
        for chosen in rejected_selected_args[k]:
            rejected_selected_args_string[k].append(",".join([str(i) for i in chosen]))
        for chosen in chosen_unselected_args[k]:
            chosen_unselected_args_string[k].append(",".join([str(i) for i in chosen]))
        for chosen in rejected_unselected_args[k]:
            rejected_unselected_args_string[k].append(",".join([str(i) for i in chosen]))

        for chosen in chosen_selected_ids[k]:
            chosen_selected_ids_string[k].append(",".join([str(i) for i in chosen]))
        for chosen in rejected_selected_ids[k]:
            rejected_selected_ids_string[k].append(",".join([str(i) for i in chosen]))
        for chosen in chosen_unselected_ids[k]:
            chosen_unselected_ids_string[k].append(",".join([str(i) for i in chosen]))
        for chosen in rejected_unselected_ids[k]:
            rejected_unselected_ids_string[k].append(",".join([str(i) for i in chosen]))

    for chosen in chosen_responses_ids:
        chosen_responses_ids_string.append(",".join([str(int(i)) for i in chosen]))
    for chosen in rejected_responses_ids:
        rejected_responses_ids_string.append(",".join([str(int(i)) for i in chosen]))

    for k in k_perc:
        results = {"prompt": o_prompts, "chosen_response": chosen_responses, "rejected_response": rejected_responses,
                   "chosen_selected_args": chosen_selected_args_string[k],
                   "rejected_selected_args": rejected_selected_args_string[k],
                   "chosen_unselected_args": chosen_unselected_args_string[k],
                   "rejected_unselected_args": rejected_unselected_args_string[k],
                   "chosen_selected_ids": chosen_selected_ids_string[k],
                   "rejected_selected_ids": rejected_selected_ids_string[k],
                   "chosen_unselected_ids": chosen_unselected_ids_string[k],
                   "rejected_unselected_ids": rejected_unselected_ids_string[k],
                   "chosen_responses_ids": chosen_responses_ids_string,
                   "rejected_responses_ids": rejected_responses_ids_string
                   }
        if os.path.exists(save_file_name):
            os.makedirs(save_file_name)
        output = pd.DataFrame(results, index=None)
        output.to_csv(os.path.join(save_file_name, "top_{}.csv".format(k)), index=False, escapechar='\\')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./tinyllama-ultrafeedback-selected-data")
    parser.add_argument("--dpo_model_path", type=str, default="./Tinyllama-chat")
    parser.add_argument("--ref_model_path", type=str, default="./tinyllama-sft/sft_hf_global_step1400")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--k_perc", type=str, default="0.3")

    args = parser.parse_args()
    args = vars(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args['device'] = device
    main(args)