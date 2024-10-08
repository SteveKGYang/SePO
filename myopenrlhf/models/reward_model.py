from typing import Optional

import torch
import torch.nn as nn
from optimum.bettertransformer import BetterTransformer
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.deepspeed import HfDeepSpeedConfig


class MentalRewardModel(nn.Module):
    """
    Reward model base class.

    Args:
        model (nn.Module): Reward model.
        value_head (nn.Module): Value head to get reward score.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrain_or_model: str,
        from_config=False,
        normalize_reward=True,
        use_flash_attention_2=False,
        to_bettertransformer=False,
        bf16=False,
    ) -> None:
        super().__init__()
        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Patch for https://github.com/huggingface/transformers/issues/28052
            def _autoset_attn_implementation_monkeypatch(cls, config, *args, **kwargs):  # type: ignore
                config._attn_implementation = attn_implementation
                return config

            PreTrainedModel._autoset_attn_implementation = classmethod(_autoset_attn_implementation_monkeypatch)

            if from_config:
                config = AutoConfig.from_pretrained(
                    pretrain_or_model,
                    torch_dtype=torch.bfloat16 if bf16 else "auto",
                    trust_remote_code=True,
                )
                self.model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrain_or_model,
                    torch_dtype=torch.bfloat16 if bf16 else "auto",
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                )


            if to_bettertransformer:
                self.model = self.model.to_bettertransformer()

            if hasattr(self.model, "transformer"):
                self.model = self.model.transformer
            elif hasattr(self.model, "model"):
                self.model = self.model.model
        else:
            self.model = pretrain_or_model

        # value head
        self.cor_value_head = nn.Linear(self.model.config.hidden_size, 1)
        self.cor_value_head.weight.data.normal_(mean=0.0, std=1 / (self.model.config.hidden_size + 1))

        self.info_value_head = nn.Linear(self.model.config.hidden_size, 1)
        self.info_value_head.weight.data.normal_(mean=0.0, std=1 / (self.model.config.hidden_size + 1))

        self.pro_value_head = nn.Linear(self.model.config.hidden_size, 1)
        self.pro_value_head.weight.data.normal_(mean=0.0, std=1 / (self.model.config.hidden_size + 1))

        # mean std
        self.normalize_reward = normalize_reward
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))

    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None):
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs["last_hidden_state"]
        cor_values = self.cor_value_head(last_hidden_states).squeeze(-1)
        info_values = self.info_value_head(last_hidden_states).squeeze(-1)
        pro_values = self.pro_value_head(last_hidden_states).squeeze(-1)

        # left padding in training mode
        if self.training:
            cor_reward = cor_values[:, -1]
            info_reward = info_values[:, -1]
            pro_reward = pro_values[:, -1]
        else:
            # The following code is equivalent to:
            #
            # last_value = []
            # for i in range(sequences.size(0)):
            #     for t in reversed(range(sequences.size(1))):
            #         if attention_mask[i][t] > 0.5:
            #             last_value.append(values[i][t])
            #             break
            # reward = torch.stack(last_value)
            #
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            cor_reward = cor_values.gather(dim=1, index=eos_indices).squeeze(1)
            info_reward = info_values.gather(dim=1, index=eos_indices).squeeze(1)
            pro_reward = pro_values.gather(dim=1, index=eos_indices).squeeze(1)

            # normalize reward in eval mode
            if self.normalize_reward:
                cor_reward = (cor_reward - self.mean) / self.std
                info_reward = (info_reward - self.mean) / self.std
                pro_reward = (pro_reward - self.mean) / self.std
        return cor_reward, info_reward, pro_reward

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def lora_enable(self, lora_rank=0, lora_train_bias="none"):
        if lora_rank > 0:
            lora_config = LoraConfig(
                inference_mode=False,
                r=lora_rank,
                lora_alpha=16,
                lora_dropout=0.05,
                bias=lora_train_bias,
            )
            self.model = get_peft_model(self.model, lora_config)


class RewardModel(nn.Module):
    """
    Reward model base class.

    Args:
        model (nn.Module): Reward model.
        value_head (nn.Module): Value head to get reward score.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrain_or_model: str,
        from_config=False,
        normalize_reward=True,
        use_flash_attention_2=False,
        bf16=False,
        ds_config=None,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Patch for https://github.com/huggingface/transformers/issues/28052
            def _autoset_attn_implementation_monkeypatch(cls, config, *args, **kwargs):  # type: ignore
                config._attn_implementation = attn_implementation
                return config

            PreTrainedModel._autoset_attn_implementation = classmethod(_autoset_attn_implementation_monkeypatch)

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if from_config:
                config = AutoConfig.from_pretrained(
                    pretrain_or_model,
                    torch_dtype=torch.bfloat16 if bf16 else "auto",
                    trust_remote_code=True,
                )
                self.model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrain_or_model,
                    torch_dtype=torch.bfloat16 if bf16 else "auto",
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                )

            if hasattr(self.model, "transformer"):
                self.model = self.model.transformer
            elif hasattr(self.model, "model"):
                self.model = self.model.model
        else:
            self.model = pretrain_or_model

        # value head
        self.value_head = nn.Linear(self.model.config.hidden_size, 1, device="cuda")
        self.value_head.weight.data.normal_(mean=0.0, std=1 / (self.model.config.hidden_size + 1))

        # mean std
        self.normalize_reward = normalize_reward
        self.register_buffer("mean", torch.zeros(1, device="cuda"))
        self.register_buffer("std", torch.ones(1, device="cuda"))

    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs["last_hidden_state"]
        values = self.value_head(last_hidden_states).squeeze(-1)

        # left padding in training mode
        if self.training:
            reward = values[:, -1]
        else:
            # The following code is equivalent to:
            #
            # last_value = []
            # for i in range(sequences.size(0)):
            #     for t in reversed(range(sequences.size(1))):
            #         if attention_mask[i][t] > 0.5:
            #             last_value.append(values[i][t])
            #             break
            # reward = torch.stack(last_value)
            #
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            # normalize reward in eval mode
            if self.normalize_reward:
                reward = (reward - self.mean) / self.std
        return reward

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def to_bettertransformer(self):
        self.model = BetterTransformer.transform(self.model)

    def reverse_bettertransformer(self):
        self.model = BetterTransformer.reverse(self.model)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def lora_enable(self, lora_rank=0, lora_train_bias="none"):
        if lora_rank > 0:
            lora_config = LoraConfig(
                inference_mode=False,
                r=lora_rank,
                lora_alpha=16,
                lora_dropout=0.05,
                bias=lora_train_bias,
            )
            self.model = get_peft_model(self.model, lora_config)