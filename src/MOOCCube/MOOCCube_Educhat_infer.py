import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("/home/ubuntu/llm_rec/models/educhat-sft-002-7b-decrypt")
model = LlamaForCausalLM.from_pretrained("/home/ubuntu/llm_rec/models/educhat-sft-002-7b-decrypt",torch_dtype=torch.float16,).half().cuda()
model = model.eval()

query = "<|prompter|>你好</s><|assistant|>"
inputs = tokenizer(query, return_tensors="pt", padding=True).to(0)
outputs = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.8, repetition_penalty=1.02, max_new_tokens=256)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)

separator = "-------------------"
prefer_info = \
'''
["世界历史", "免疫学", "农学", "力学", "化学", "医学", "地理学", "地质学", "建筑学", "心理学", "教育学", "数学", "机械工程", "材料科学技术", "物理学", "电子学", "电气工程", "管理科学技术", "自然辩证法", "航天科学技术", "航空科学技术", "船舶工程", "计算机科学技术", "语言学"]
'''
basic_prompt = "您需要预测学生未来选课的偏好，从偏好列表中挑选即可。偏好列表如下：" + prefer_info
system_prompt = "<|system|>" + basic_prompt + "</s>"
quest = system_prompt + "<|prompter|>同学0历史记录: ['C++语言程序设计进阶', '数据结构(上)', 'C++语言程序设计基础', '线性代数(1)', '数据结构(下)']</s><|assistant|>"
reply = '''同学0选课偏好: ["计算机科学技术", "数学", "电子学", "电气工程"]</s>'''
example_prompt = quest + reply

cnt = 0
with open("/home/ubuntu/llm_rec/output/output_educhat.txt", "w", encoding='utf-8') as f_out:
    with open("/home/ubuntu/llm_rec/prompts/prompt_info.txt", "r", encoding='utf-8') as f_in:
        for line in f_in:
            if cnt % 100 == 0:
                print(cnt)
            user_info = line.strip()
            query = example_prompt + "<|prompter|>" + user_info + "</s><|assistant|>"
            inputs = tokenizer(query, return_tensors="pt", padding=True).to(0)
            outputs = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.8, repetition_penalty=1.10, max_new_tokens=512)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(cnt, file=f_out)
            print(response, file=f_out)
            print(separator, file=f_out)
            cnt += 1
            if cnt <= 10:
                print(query)
                print(response)
                print(separator)
