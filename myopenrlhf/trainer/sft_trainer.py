import math
import os.path
from abc import ABC

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from transformers.trainer import get_scheduler

from myopenrlhf.datasets import SFTDataset
from myopenrlhf.models.loss import GPTLMLoss, RRHFLoss


class RRHFTrainer(ABC):
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
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.gradient_checkpointing = gradient_checkpointing

        # misc
        #self.loss_fn = RRHFLoss()
        self.loss_fn = GPTLMLoss()

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # wandb setting
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
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            for prompts_id_len, chosen_inputs, chosen_attention_masks, reject_inputs, reject_attention_masks, _ in self.train_dataloader:
                #print(torch.cuda.current_device())
                print(chosen_inputs.shape)
                chosen_inputs = chosen_inputs.squeeze(1).to(torch.cuda.current_device())
                chosen_attention_mask = chosen_attention_masks.squeeze(1).to(torch.cuda.current_device())

                # reject_inputs = reject_inputs.squeeze(1).to(torch.cuda.current_device())
                # reject_attention_mask = reject_attention_masks.squeeze(1).to(torch.cuda.current_device())
                #
                # logits, labels = self.concatenated_forward(
                #     self.model, chosen_inputs, chosen_attention_mask, reject_inputs, reject_attention_mask,
                #     prompts_id_len
                # )
                #
                # loss = self.loss_fn(logits, labels)

                logits = self.model(chosen_inputs, attention_mask=chosen_attention_mask, return_output=True)["logits"]

                labels = torch.where(
                    chosen_attention_mask.bool(),
                    chosen_inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                if not self.pretrain_mode:
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(logits, labels)

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                logs_dict = {
                    "train_loss": loss.item(),
                }
                # logs/checkpoints/evaluation
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1

            epoch_bar.update()

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, prompts_id_len):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, chosen_input_ids, chosen_att_masks, reject_input_ids, reject_att_masks\
            = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        logits = model(input_ids, attention_mask=att_masks, return_output=True)["logits"]
        # chosen_logits = logits[: chosen_ids.shape[0]]
        # rejected_logits = logits[chosen_ids.shape[0] :]

        # chosen_labels = torch.where(
        #     chosen_att_masks.bool(),
        #     chosen_input_ids,
        #     self.loss_fn.IGNORE_INDEX,
        # )
        labels = torch.where(
            att_masks.bool(),
            input_ids,
            self.loss_fn.IGNORE_INDEX,
        )
        if not self.pretrain_mode:
            # for label, source_len in zip(chosen_labels, prompts_id_len):
            #     label[:source_len] = self.loss_fn.IGNORE_INDEX
            for label, source_len in zip(labels, prompts_id_len):
                label[:source_len] = self.loss_fn.IGNORE_INDEX

        return logits, labels

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
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])

        chosen_input_ids = pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id)
        reject_input_ids = pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id)
        inputs_ids = torch.cat(
            (
                chosen_input_ids,
                reject_input_ids,
            ),
            dim=0,
        )

        max_length = max(c_mask.shape[1], r_mask.shape[1])
        chosen_att_masks = pad_to_length(c_mask, max_length, 0)
        reject_att_masks = pad_to_length(r_mask, max_length, 0)
        att_masks = torch.cat((chosen_att_masks, reject_att_masks), dim=0)
        return inputs_ids, att_masks, chosen_input_ids, chosen_att_masks, reject_input_ids, reject_att_masks

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
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
            self.model.save_pretrained(args.save_path, global_step, args.aspect)
            #self.strategy.save_hf_format(self.model.model, self.tokenizer, args.save_path)

    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompts_id_len, chosen_inputs, chosen_attention_masks, reject_inputs, reject_attention_masks, _ in self.train_dataloader:
                chosen_inputs = chosen_inputs.squeeze(1).to(torch.cuda.current_device())
                chosen_attention_mask = chosen_attention_masks.squeeze(1).to(torch.cuda.current_device())

                # reject_inputs = reject_inputs.squeeze(1).to(torch.cuda.current_device())
                # reject_attention_mask = reject_attention_masks.squeeze(1).to(torch.cuda.current_device())
                #
                # logits, labels = self.concatenated_forward(
                #     self.model, chosen_inputs, chosen_attention_mask, reject_inputs, reject_attention_mask,
                #     prompts_id_len
                # )
                #
                # loss = self.loss_fn(logits, labels)

                logits = self.model(chosen_inputs, attention_mask=chosen_attention_mask, return_output=True)["logits"]

                labels = torch.where(
                    chosen_attention_mask.bool(),
                    chosen_inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                if not self.pretrain_mode:
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            '''for prompts_id_len, inputs, attention_masks, _ in eval_dataloader:
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                logits = self.model(inputs, attention_mask=attention_mask, return_output=True)["logits"]

                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                if not self.pretrain_mode:
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                loss = self.loss_fn(logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)'''

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state


class SFTTrainer(ABC):
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
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.gradient_checkpointing = gradient_checkpointing

        # misc
        self.loss_fn = GPTLMLoss()

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # wandb setting
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
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            for prompts_id_len, inputs, attention_masks, _ in self.train_dataloader:
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                logits = self.model(inputs, attention_mask=attention_mask, return_output=True)["logits"]

                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                if not self.pretrain_mode:
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(logits, labels)
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                logs_dict = {
                    "train_loss": loss.item(),
                }
                # logs/checkpoints/evaluation
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1

            epoch_bar.update()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
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
                args.save_path + "/sft_hf_{}".format(tag),
            )

    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompts_id_len, inputs, attention_masks, _ in eval_dataloader:
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                logits = self.model(inputs, attention_mask=attention_mask, return_output=True)["logits"]

                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                if not self.pretrain_mode:
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                loss = self.loss_fn(logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state
