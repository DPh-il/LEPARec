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
['创业', '电子', '工程', '公共管理', '化学', '环境·地球', '计算机', '建筑', '教育', '经管·会计', '历史', '汽车', '社科·法律', '生命科学', '数学', '外语', '文学', '物理', '护理学', '艺术·设计', '哲学', '职场', '其他']
'''
basic_prompt = "您需要预测学生未来选课的偏好，从偏好列表中挑选即可。偏好列表如下：" + prefer_info
system_prompt = "<|system|>" + basic_prompt + "</s>"
quest = system_prompt + "<|prompter|>同学0历史记录: ['中国建筑史（上）', '外国工艺美术史', '心理学概论', '经济学原理', '公司金融', '创业102：你能为客户做什么？', 'e时代的教与学――MOOC引发的混合式教学', '党的十九大精神概论']</s><|assistant|>"
reply = '''同学0选课偏好:  ['创业', '计算机', '建筑', '艺术·设计', '教育', '经管·会计', '社科·法律', '其他']</s>'''
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
