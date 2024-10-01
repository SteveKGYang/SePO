<h1 align="center">Selective Preference Optimization via Token-Level Reward Function Estimation </h1>

## Contents
- [Introduction](#introduction)
- [Preparation](#preparation)
- [Model Weights](#sepo-models )
- [Token Selection](#token-selection)
- [Model Training](#model-training)

## Introduction
Recent advancements in large language model alignment leverage token-level supervisions to perform fine-grained
preference optimization. However, existing token-level alignment methods either optimize on all available tokens, 
which can be noisy and inefficient, or perform selective training with complex and expensive key token selection 
strategies. In this work, we propose Selective Preference Optimization (SePO), the first selective alignment strategy
that centers on efficient key token selection without requiring strong, fine-grained supervision signals. SePO trains
an oracle model with direct preference optimization (DPO) to estimate a token-level reward function on the target data,
which applies to any existing alignment datasets with response-level annotations and enables cost-efficient token
selection with small-scale oracle models and training data. The estimated reward function is then utilized to score all
tokens within the target dataset, where only the key tokens are selected to supervise the target policy model with a 
reference model-free contrastive objective function. Extensive experiments on three public evaluation benchmarks show
that SePO significantly outperforms competitive baseline methods by only optimizing on 30% key tokens. Further 
applications on weak-to-strong generalization show that weak oracle models effectively supervise strong policy models
with up to 16.8x more parameters. SePO can also select useful training signals from out-of-distribution data to improve
policy models, alleviating the over-optimization problem.

## Preparation

1. Set up the Python 3.10 environment.
2. Build the dependencies with the following code:
```bash
pip install -r requirements.txt
```

## SePO Models 

We provide the TinyLLaMA-based oracle-sft model pair for estimating the token-level reward function, and 
3 LLaMA policy models that are selectively fine-tuned and evaluated in the SePO paper.

### Oracle Model Checkpoints
- [Ref-Tinyllama](https://huggingface.co/SePO2024/Ref-Tinyllama): 
This model is fine-tuned based on the TinyLLaMA foundation model and the full UltraChat-200K dataset.
The training paradigm is supervised fine-tuning. This model is used as the reference model for estimating the
token-level reward function.

- [Oracle-Tinyllama](https://huggingface.co/SePO2024/Oracle-Tinyllama): 
This model is fine-tuned based on the above Ref-Tinyllama model and the full UltraFeedback dataset.
The training paradigm is DPO. This model is used as the oracle model for estimating the
token-level reward function.

### Policy Model Checkpoints
- [SePO-tinyllama-chat](https://huggingface.co/SePO2024/SePO-tinyllama-chat): 
This model is fine-tuned based on the TinyLLaMA-Chat foundation model and the selective training 
dataset built from the TinyLLaMA-based oracle-sft model pair trained on the full UltraChat-200K and Ultrafeedback dataset.
The Top-30% tokens were selected to train the policy model.
The model is trained to provide general responses of an AI assistant considering a single-turn query, but the queries include
professional questions such as programming language and history, and the aligned responses are usually quite
complicated. The model is also not especially optimized for complicated reasoning tasks.

- [SePO-llama2-chat-7b](https://huggingface.co/SePO2024/SePO-llama2-chat-7b): 
This model is fine-tuned based on the LLaMA2-Chat-7B foundation model and the selective training 
dataset built from the TinyLLaMA-based oracle-sft model pair trained on the full UltraChat-200K and Ultrafeedback dataset.
The Top-30% tokens were selected to train the policy model.
The model is trained to provide general responses of an AI assistant considering a single-turn query, but the queries include
professional questions such as programming language and history, and the aligned responses are usually quite
complicated. The model is also not especially optimized for complicated reasoning tasks.

- [SePO-llama2-chat-13b](https://huggingface.co/SePO2024/SePO-llama2-chat-13b): 
This model is fine-tuned based on the LLaMA2-Chat-7B foundation model and the selective training 
dataset built from the TinyLLaMA-based oracle-sft model pair trained on the full UltraChat-200K and Ultrafeedback dataset.
The Top-30% tokens were selected to train the policy model.
The model is trained to provide general responses of an AI assistant considering a single-turn query, but the queries include
professional questions such as programming language and history, and the aligned responses are usually quite
complicated. The model is also not especially optimized for complicated reasoning tasks.

### Inference
With the Hugging Face Transformers library, you can use the SePO models in your Python project. Here is a simple example of how to load the model:

```python
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
tokenizer = LlamaTokenizer.from_pretrained(PATH_TO_MODEL, padding_side='left')
model = LlamaForCausalLM.from_pretrained(PATH_TO_TOKENIZER, device_map='auto', torch_dtype=torch.bfloat16)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

In this example, LlamaTokenizer is used to load the tokenizer, and LlamaForCausalLM is used to load the model. The `device_map='auto'` argument is used to automatically
use the GPU if it's available.

You can use the trained policy model as an AI assistant. 
To ensure the best performance, use the corresponding template to prompt the trained policy models:

```python
query = 'What are the names of some famous actors that started their careers on Broadway?'
templated_query = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```
You can obtain a response from the policy model using the following codes:

```python
inputs = tokenizer(templated_query, return_tensors="pt", padding=True)
input_ids = inputs.input_ids.to(device)
generate_ids = model.generate(input_ids, max_new_tokens=1024)
truc_ids = generate_ids[0][len(input_ids[0]):]
response = tokenizer.decode(truc_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
print(response)
```

For example, one inference of SePO-llama2-chat-13b on the above codes has the following response:
```
Many famous actors have gotten their start on Broadway before going on to successful careers in film and television. Here are some examples:\n\n1. Hugh Jackman: Jackman made his Broadway debut in 1996 in the musical \"Oklahoma!\" and went on to star in other productions such as \"The Boy from Oz\" and \"Les Mis\u00e9rables\" before breaking into film with roles in \"X-Men\" and \"Kate & Leopold.\"\n2. Anne Hathaway: Hathaway made her Broadway debut in 2002 in the play \"The Last Night of Ballyhoo\" and went on to star in \"Les Mis\u00e9rables\" and \"Rachel Getting Married\" before transitioning to film with roles in \"The Devil Wears Prada\" and \"Les Mis\u00e9rables.\"\n3. Jake Gyllenhaal: Gyllenhaal made his Broadway debut in 1998 in the play \"The Rainmaker\" and went on to star in \"The Pillowman\" and \"If There Is I Haven't Found It Yet\" before breaking into film with roles in \"Donnie Darko\" and \"Brokeback Mountain.\"\n4. Idina Menzel: Menzel made her Broadway debut in 1995 in the musical \"Rent\" and went on to star in \"Wicked\" and \"If\/Then\" before breaking into film with roles in \"Enchanted\" and \"Frozen.\"\n5. Lin-Manuel Miranda: Miranda made his Broadway debut in 2008 in the musical \"In the Heights\" and went on to create and star in the hit musical \"Hamilton,\" for which he won a Pulitzer Prize and multiple Tony Awards. He has also appeared in films such as \"Mary Poppins Returns\" and \"Moana.\"\n6. Ben Platt: Platt made his Broadway debut in 2016 in the musical \"Dear Evan Hansen\" and won a Tony Award for his performance. He has also appeared in films such as \"Pitch Perfect\" and \"The Politician.\"\n7. Laura Linney: Linney made her Broadway debut in 1986 in the play \"The Crucible\" and has since appeared in numerous productions on the Great White Way, including \"Sight Unseen\" and \"The Little Foxes.\" She has also had a successful film career, with roles in \"The Big C\" and \"Ozark.\"\n8. Jude Law: Law made his Broadway debut in 1996 in the play \"The Idiot\" and has since appeared in productions such as \"Hamlet\" and \"The Vote.\" He has also had a successful film career, with roles in \"The Talented Mr. Ripley\" and \"Fantastic Beasts: The Crimes of Grindelwald.\"\n9. Cynthia Erivo: Erivo made her Broadway debut in 2015 in the musical \"The Color Purple\" and won a Tony Award for her performance. She has also appeared in films such as \"Widows\" and \"Harriet.\"\n10. Bryan Cranston: Cranston made his Broadway debut in 1988 in the play \"The God of Hell\" and has since appeared in productions such as \"All the Way\" and \"Network.\" He has also had a successful film career, with roles in \"Breaking Bad\" and \"Trumbo.\"\n\nThese are just a few examples of actors who got their start on Broadway before going on to successful careers in film and television. Many other actors have also made their Broadway debuts before transitioning to other mediums, including Meryl Streep, Denzel Washington, and Emma Stone.
```

## Token Selection
You can use our existing codes and provided oracle-ref mode pair to estimate the token-level reward function
and perform token selection on your own data. We provide a script in ```token_score.py```, which can be used to select
for the [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) dataset. You can use the following
command to execute the code:
```bash
python token_score.py --save_dir SAVE_DIR --dpo_model_path ORACLE_MODEL_PATH --ref_model_path REF_MODEL_PATH --k_perc PERCENTAGES
```
where ```PERCENTAGES``` denotes the token selection percentages that you want to obtain, separated by ",". For example,
if you want to obtain percentages 0.1, 0.3, and 0.5, it can be written as 0.1,0.3,0.5. The selected tokens for each percentage
is stored in a separate file. All files will be stored in ```SAVE_DIR```.

## Model Training
### Oracle Model Training
The reference model is trained based on an SFT manner with the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) framework.
If you hope to re-train the model, set up the OpenRLHF framework according to their guidelines and fine-tune the model
with new data. Modify the arguments in train_sft_llama.sh to customize your own settings. The following command is an example of fine-tuning the reference model:
```bash
bash train_sft_llama.sh
```

The oracle model is trained based on an DPO manner with the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) framework.
If you hope to re-train the model, set up the OpenRLHF framework according to their guidelines and fine-tune the model
with new data. Modify the arguments in train_dpo_llama.sh to customize your own settings. The following command is an example of fine-tuning the oracle model:
```bash
bash train_dpo_llama.sh
```

We ran the above codes on 2 x Nvidia A100 80GB GPUs. You can modify the settings to fit your own needs.

### SePO Training
The policy models are trained based on the SePO algorithm and the selective dataset, with the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) framework.
If you hope to train a new policy model, set up the OpenRLHF framework according to their guidelines and fine-tune the model
with new data. Modify the arguments in train_selective_llama.sh to customize your own settings. The following command is an example of fine-tuning the SePO model:
```bash
bash train_selective_llama.sh
```