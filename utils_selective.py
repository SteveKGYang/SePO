from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from abc import ABC
import pandas as pd
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from myopenrlhf.datasets.utils import exist_and_not_none, zero_pad_sequences


class SelectiveRRHFLoss(nn.Module):
    """
    RRHF Language Model Loss
    """

    def __init__(self):
        super().__init__()

    def RRHF(self, chosen_reward, reject_reward):
        return F.relu(reject_reward - chosen_reward)

    def SimPO(self, chosen_reward, reject_reward):
        return -F.logsigmoid(chosen_reward-reject_reward)

    def pad_to_length(self, tensor, length, pad_value, dim=-1):
        if tensor.size(dim) >= length:
            return tensor
        else:
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            return torch.cat(
                [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
            )

    def forward(self, chosen_logits: torch.Tensor, reject_logits: torch.Tensor,
                chosen_masks: torch.Tensor, rejected_masks: torch.Tensor, prompt_mask: torch.Tensor):
        chosen_masks = self.pad_to_length(chosen_masks[:, :-1], chosen_logits.shape[-1], 0)
        rejected_masks = self.pad_to_length(rejected_masks[:, :-1], chosen_logits.shape[-1], 0)
        # prompt_mask = self.pad_to_length(prompt_mask, chosen_logits.shape[-1], 0)

        chosen_rrhf_reward = (chosen_logits.cpu() * chosen_masks).sum(-1)/ chosen_masks.sum(-1)
        reject_rrhf_reward = (reject_logits.cpu() * rejected_masks).sum(-1) / rejected_masks.sum(-1)

        chosen_rrhf_reward = chosen_rrhf_reward.to(torch.cuda.current_device())
        reject_rrhf_reward = reject_rrhf_reward.to(torch.cuda.current_device())
        # prompt_mask = 1 - prompt_mask
        # sft_loss = -(chosen_logits * prompt_mask).sum(-1)/prompt_mask.sum(-1)
        #rrhf_loss = self.RRHF(chosen_rrhf_reward, reject_rrhf_reward)
        rrhf_loss = self.SimPO(chosen_rrhf_reward, reject_rrhf_reward)

        return torch.mean(rrhf_loss), torch.mean(chosen_rrhf_reward), torch.mean(reject_rrhf_reward)
        # return torch.mean(sft_loss+rrhf_loss), torch.mean(chosen_rrhf_reward), torch.mean(reject_rrhf_reward)


class SelectiveTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        strategy,
        tokenizer,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm=0.5,
        beta=0.01,
        max_epochs: int = 2,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.gradient_checkpointing = gradient_checkpointing
        self.tokenizer = tokenizer

        self.beta = beta
        #self.loss_fn = DPOLoss(self.beta, self.args.label_smoothing, self.args.ipo)
        self.loss_fn = SelectiveRRHFLoss()

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(self.epochs):
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            acc_mean = 0
            loss_mean = 0
            # train
            for chosen_ids, c_mask, reject_ids, r_mask, \
            chosen_selected_masks, chosen_unselected_masks, rejected_selected_masks, rejected_unselected_masks,\
                   prompt_masks in self.train_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                chosen_logps, rejected_logps = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask
                )
                # with torch.no_grad():
                #     reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                #         self.ref_model, chosen_ids, c_mask, reject_ids, r_mask
                #     )
                loss, chosen_reward, reject_reward = self.loss_fn(
                    chosen_logps, rejected_logps, chosen_selected_masks
                    , rejected_unselected_masks, prompt_masks.squeeze(1)
                )

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc_mean = acc_mean * 0.9 + 0.1 * (chosen_reward > reject_reward).float().mean().item()
                loss_mean = loss_mean * 0.9 + 0.1 * loss.item()
                # dpo logs
                logs_dict = {
                    "train_loss": loss.item(),
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "acc_mean": acc_mean,
                    "loss_mean": loss_mean,
                }
                # logs/checkpoints/evaluate
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        # logs
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            #self.strategy.save_ckpt(self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)
            self.strategy.save_hf_format(
            self.model.model,
            self.tokenizer,
            self.args.save_path + "/dpo_hf_{}".format(tag),
        )

    def evaluate(self, eval_dataloader, steps=0):
        self.model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of global_step %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            acc = 0
            loss_sum = 0
            for chosen_ids, c_mask, reject_ids, r_mask, \
                    chosen_selected_masks, chosen_unselected_masks, rejected_selected_masks, rejected_unselected_masks, \
                    prompt_masks in eval_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                chosen_logps, rejected_logps = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask
                )
                # with torch.no_grad():
                #     reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                #         self.ref_model, chosen_ids, c_mask, reject_ids, r_mask
                #     )
                loss, chosen_reward, reject_reward = self.loss_fn(
                    chosen_logps, rejected_logps, chosen_selected_masks
                    , rejected_unselected_masks,
                    prompt_masks.squeeze(1)
                )
                acc += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                step_bar.update()

            acc_mean = acc / self.eval_dataloader.__len__()
            loss_mean = loss_sum / self.eval_dataloader.__len__()

            logs = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
            }
            logs = self.strategy.all_reduce(logs)
            step_bar.set_postfix(logs)
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        all_logits = model(input_ids, attention_mask=att_masks, return_output=True)["logits"]
        all_logps = self._get_batch_logps(all_logits, input_ids, attention_mask=att_masks, average_log_prob=False)
        chosen_logps = all_logps[: chosen_ids.shape[0]]
        rejected_logps = all_logps[chosen_ids.shape[0] :]
        return chosen_logps, rejected_logps

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks

    def _get_batch_logps(
        self, logits: torch.FloatTensor, labels: torch.LongTensor, attention_mask, average_log_prob: bool = False
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = attention_mask[:, 1:].bool()
        # dummy token; we'll ignore the losses on these tokens later
        labels[loss_mask == False] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return per_token_logps * loss_mask


class SelectiveDataset(Dataset):
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
        split,
        tokenizer: Callable,
        max_length: int,
        strategy,
        pretrain_mode=False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.split = split

        dataset_file = pd.read_csv(dataset)
        if split == 'train':
            self.prompts = dataset_file['prompt'].to_list()[4000:]
            self.chosen_response = dataset_file['chosen_response'].to_list()[4000:]
            self.rejected_response = dataset_file['rejected_response'].to_list()[4000:]
            self.chosen_selected_args = dataset_file['chosen_selected_args'].to_list()[4000:]
            self.rejected_selected_args = dataset_file['rejected_selected_args'].to_list()[4000:]
            self.chosen_unselected_args = dataset_file['chosen_unselected_args'].to_list()[4000:]
            self.rejected_unselected_args = dataset_file['rejected_unselected_args'].to_list()[4000:]
            self.chosen_responses_ids = dataset_file['chosen_responses_ids'].to_list()[4000:]
            self.rejected_responses_ids = dataset_file['rejected_responses_ids'].to_list()[4000:]
        elif split == 'eval':
            self.prompts = dataset_file['prompt'].to_list()[:4000]
            self.chosen_response = dataset_file['chosen_response'].to_list()[:4000]
            self.rejected_response = dataset_file['rejected_response'].to_list()[:4000]
            self.chosen_selected_args = dataset_file['chosen_selected_args'].to_list()[:4000]
            self.rejected_selected_args = dataset_file['rejected_selected_args'].to_list()[:4000]
            self.chosen_unselected_args = dataset_file['chosen_unselected_args'].to_list()[:4000]
            self.rejected_unselected_args = dataset_file['rejected_unselected_args'].to_list()[:4000]
            self.chosen_responses_ids = dataset_file['chosen_responses_ids'].to_list()[:4000]
            self.rejected_responses_ids = dataset_file['rejected_responses_ids'].to_list()[:4000]
        else:
            raise ValueError("Unidentified split!")

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        #print(idx)
        prompt = self.prompts[idx]
        chosen = self.chosen_response[idx]
        rejected = self.rejected_response[idx]
        chosen_selected_arg = self.chosen_selected_args[idx]
        chosen_unselected_arg = self.chosen_unselected_args[idx]
        rejected_selected_arg = self.rejected_selected_args[idx]
        rejected_unselected_arg = self.rejected_unselected_args[idx]

        # chosen_selected_id = self.chosen_selected_ids[idx]
        # chosen_unselected_id = self.chosen_unselected_ids[idx]
        # rejected_selected_id = self.rejected_selected_ids[idx]
        # rejected_unselected_id = self.rejected_unselected_ids[idx]

        chosen_response_id = self.chosen_responses_ids[idx]
        rejected_response_id = self.rejected_responses_ids[idx]
        chosen_response_id = torch.tensor([[int(k) for k in chosen_response_id.split(",")]], dtype=torch.int)
        rejected_response_id = torch.tensor([[int(k) for k in rejected_response_id.split(",")]], dtype=torch.int)

        messages = [
                {"role": "user", "content": prompt}
            ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # chosen_tokens = self.tokenizer(
        #     chosen,
        #     max_length=self.max_length,
        #     padding=False,
        #     truncation=True,
        #     return_tensors="pt",
        # )

        chosen_selected_mask = torch.zeros([len(prompt_tokens.input_ids[0])+len(chosen_response_id[0])])
        chosen_unselected_mask = chosen_selected_mask.clone()

        for pos in chosen_selected_arg.split(","):
            chosen_selected_mask[len(prompt_tokens.input_ids[0])+int(pos)] = 1
        for pos in chosen_unselected_arg.split(","):
            chosen_unselected_mask[len(prompt_tokens.input_ids[0])+int(pos)] = 1

        chosen_ids = torch.cat([prompt_tokens.input_ids, chosen_response_id], dim=-1)
        chosen_mask = torch.ones([1, chosen_selected_mask.shape[0]])
        assert chosen_ids.shape[-1] == chosen_mask.shape[-1]

        # rejected_tokens = self.tokenizer(
        #     rejected,
        #     max_length=self.max_length,
        #     padding=False,
        #     truncation=True,
        #     return_tensors="pt",
        # )

        rejected_selected_mask = torch.zeros([len(prompt_tokens.input_ids[0]) + len(rejected_response_id[0])])
        rejected_unselected_mask = rejected_selected_mask.clone()

        for pos in rejected_selected_arg.split(","):
            rejected_selected_mask[len(prompt_tokens.input_ids[0])+int(pos)] = 1
        for pos in rejected_unselected_arg.split(","):
            rejected_unselected_mask[len(prompt_tokens.input_ids[0])+int(pos)] = 1

        rejected_ids = torch.cat([prompt_tokens.input_ids, rejected_response_id], dim=-1)
        rejected_mask = torch.ones([1, rejected_selected_mask.shape[0]])
        assert rejected_ids.shape[-1] == rejected_mask.shape[-1]

        # to avoid EOS_token truncation
        chosen_ids[0][-1] = self.tokenizer.eos_token_id
        chosen_mask[0][-1] = True

        rejected_ids[0][-1] = self.tokenizer.eos_token_id
        rejected_mask[0][-1] = True

        if self.split == 'eval':
            chosen_selected_mask = torch.ones(chosen_selected_mask.shape)
            chosen_unselected_mask = torch.ones(chosen_unselected_mask.shape)
            rejected_selected_mask = torch.ones(rejected_selected_mask.shape)
            rejected_unselected_mask = torch.ones(rejected_unselected_mask.shape)

        return chosen_ids, chosen_mask, \
            rejected_ids, rejected_mask, \
            chosen_selected_mask, chosen_unselected_mask, rejected_selected_mask, \
            rejected_unselected_mask, prompt_tokens.attention_mask
        #return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info

    def collate_fn(self, item_list):
        chosen_input_ids = []
        chosen_attention_masks = []

        rejected_input_ids = []
        rejected_attention_masks = []

        chosen_selected_masks = []
        chosen_unselected_masks = []
        rejected_selected_masks = []
        rejected_unselected_masks = []

        prompt_masks = []

        for chosen_id, chosen_mask, rejected_id, rejected_mask, chosen_selected_mask, chosen_unselected_mask, rejected_selected_mask, \
            rejected_unselected_mask, prompt_mask in item_list:
            chosen_input_ids.append(chosen_id)
            chosen_attention_masks.append(chosen_mask)
            rejected_input_ids.append(rejected_id)
            rejected_attention_masks.append(rejected_mask)

            chosen_selected_masks.append(chosen_selected_mask)
            chosen_unselected_masks.append(chosen_unselected_mask)
            rejected_selected_masks.append(rejected_selected_mask)
            rejected_unselected_masks.append(rejected_unselected_mask)
            prompt_masks.append(prompt_mask)

        chosen_input_ids = zero_pad_sequences(chosen_input_ids, "left", self.tokenizer.pad_token_id)
        chosen_attention_masks = zero_pad_sequences(chosen_attention_masks, "left")
        rejected_input_ids = zero_pad_sequences(rejected_input_ids, "left", self.tokenizer.pad_token_id)
        rejected_attention_masks = zero_pad_sequences(rejected_attention_masks, "left")

        chosen_selected_masks = zero_pad_sequences(chosen_selected_masks, "left")
        chosen_unselected_masks = zero_pad_sequences(chosen_unselected_masks, "left")
        rejected_selected_masks = zero_pad_sequences(rejected_selected_masks, "left")
        rejected_unselected_masks = zero_pad_sequences(rejected_unselected_masks, "left")
        prompt_masks = zero_pad_sequences(prompt_masks, "right")

        return chosen_input_ids, chosen_attention_masks, rejected_input_ids, rejected_attention_masks, \
            chosen_selected_masks, chosen_unselected_masks, rejected_selected_masks, rejected_unselected_masks, \
            prompt_masks

# tokenizer = AutoTokenizer.from_pretrained("/mnt/iusers01/nactem01/g36374ky/scratch/FastChat/llama2-chat-7B")
# #tokenizer = AutoTokenizer.from_pretrained("./Tinyllama-chat")
# dataset = SelectiveDataset("./toy_ultrafeedback_selected_args.csv", "eval", tokenizer, 2048, None)
# item_list = [dataset.__getitem__(i) for i in range(len(dataset.prompts))]
# #dataset.__getitem__(55)
# print("Done")
