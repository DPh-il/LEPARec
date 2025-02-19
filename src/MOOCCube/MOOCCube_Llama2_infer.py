# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
import json

import torch
from transformers import LlamaTokenizer
import torch.nn as nn

from llama_recipes.inference.model_utils import load_model, load_peft_model


def main(
        model_name,
        peft_model: str = None,
        quantization: bool = False,
        max_new_tokens=256,  # The maximum numbers of tokens to generate
        prompt_file: str = None,
        seed: int = 42,  # seed value for reproducibility
        do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
        min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
        use_cache: bool = True,
        # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        top_p: float = 1.0,
        # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
        top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty: int = 1,
        # [optional] Exponential penalty to the length that is used with beam-based generation.
        enable_azure_content_safety: bool = False,  # Enable safety check with Azure content safety api
        enable_sensitive_topics: bool = False,  # Enable check for sensitive topics using AuditNLG APIs
        enable_salesforce_content_safety: bool = True,  # Enable safety check with Salesforce safety flan t5
        max_padding_length: int = None,  # the max padding length to be used with tokenizer padding the prompts.
        use_fast_kernels: bool = False,
        # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
        **kwargs
):
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    if use_fast_kernels:
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dirs = "/home/ubuntu/llm_rec/llama_recipes/output/"
    output_path = os.path.join(dirs, 'user_text_only_{}.txt'.format(time.time()))

    weight = model.state_dict()['model.embed_tokens.weight']
    llama2_embLayer = nn.Embedding.from_pretrained(weight)

    separator = "-------------------"
    prefer_info = \
        '''
        ["世界历史", "免疫学", "农学", "力学", "化学", "医学", "地理学", "地质学", "建筑学", "心理学", "教育学", "数学", "机械工程", "材料科学技术", "物理学", "电子学", "电气工程", "管理科学技术", "自然辩证法", "航天科学技术", "航空科学技术", "船舶工程", "计算机科学技术", "语言学"]
        '''
    basic_prompt = "您需要预测学生未来选课的偏好，从偏好列表中挑选即可。偏好列表如下：" + prefer_info
    system_prompt = "<s>[INST]<<SYS>>" + basic_prompt + "<</SYS>>"
    quest = system_prompt + "同学0历史记录: ['C++语言程序设计进阶', '数据结构(上)', 'C++语言程序设计基础', '线性代数(1)', '数据结构(下)']" + "[/INST]"
    reply = '''同学0选课偏好: ["计算机科学技术", "数学", "电子学", "电气工程"]</s>'''
    example_prompt = quest + reply

    cnt = 0
    with open("/home/ubuntu/llm_rec/output/output_llama2.txt", "w", encoding='utf-8') as f_out:
        with open("/home/ubuntu/llm_rec/prompts/prompt_info.txt", "r", encoding='utf-8') as f_in:
            for line in f_in:
                if cnt % 100 == 0:
                    print(cnt)
                user_info = line.strip()
                query = example_prompt + "<s>[INST]" + user_info + "[/INST]"
                inputs = tokenizer(query, padding='max_length', truncation=True, max_length=max_padding_length,
                                  return_tensors="pt")
                batch = {k: v.to("cuda") for k, v in inputs.items()}
                # print(len(batch['input_ids'][0]))
                inputs_embeds = llama2_embLayer(batch['input_ids'])
                start = time.perf_counter()

                with torch.no_grad():
                    outputs = model.generate(
                        # **batch,
                        attention_mask=batch['attention_mask'],
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        top_p=top_p,
                        temperature=temperature,
                        min_length=min_length,
                        use_cache=use_cache,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        **kwargs
                    )

                e2e_inference_time = (time.perf_counter() - start) * 1000
                # print(f"the inference time is {e2e_inference_time} ms")

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(cnt, file=f_out)
                print(response, file=f_out)
                print(separator, file=f_out)
                if cnt <= 10:
                    print(cnt)
                    print(query)
                    print(response)
                    print(separator)
                cnt += 1


if __name__ == "__main__":
    fire.Fire(main)
