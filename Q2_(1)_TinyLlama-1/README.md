---
license: Apache License 2.0
language:
  - en
tasks:
  - text-generation
---

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">

# TinyLlama-1.1B
</div>

https://github.com/jzhang38/TinyLlama

The TinyLlama project aims to **pretrain** a **1.1B Llama model on 3 trillion tokens**. With some proper optimization, we can achieve this within a span of "just" 90 days using 16 A100-40G GPUs 🚀🚀. The training has started on 2023-09-01. 

<div align="center">
  <img src="./TinyLlama_logo.png" width="300"/>
</div>

We adopted exactly the same architecture and tokenizer as Llama 2. This means TinyLlama can be plugged and played in many open-source projects built upon Llama. Besides, TinyLlama is compact with only 1.1B parameters. This compactness allows it to cater to a multitude of applications demanding a restricted computation and memory footprint.

#### This Model
This is the chat model finetuned on [PY007/TinyLlama-1.1B-intermediate-step-240k-503b](https://huggingface.co/PY007/TinyLlama-1.1B-intermediate-step-240k-503b). The dataset used is [openassistant-guananco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco).

#### How to use
You will need the transformers>=4.31
Do check the [TinyLlama](https://github.com/jzhang38/TinyLlama) github page for more information.
```
from transformers import AutoTokenizer
import transformers 
import torch
model = "PY007/TinyLlama-1.1B-Chat-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "What are the values in open source projects?"
formatted_prompt = (
    f"### Human: {prompt} ### Assistant:"
)


sequences = pipeline(
    formatted_prompt,
    do_sample=True,
    top_k=50,
    top_p = 0.7,
    num_return_sequences=1,
    repetition_penalty=1.1,
    max_new_tokens=500,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
```