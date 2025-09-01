
import os
import argparse
import openai
import time

openai.api_key = "..."

openai.base_url = "https://api.v3.cm/v1/"
openai.default_headers = {"x-foo": "true"}

def process_prompt(system_msg, example_msg, prompt_text):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": example_msg},
        {"role": "user", "content": prompt_text}
    ]
    
    response = openai.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )
    
    return response.choices[0].message.content.strip()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量调用OpenAI API处理prompt')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--prompt_file', default='prompt.txt')
    parser.add_argument('--output_file', default='output.txt')
    parser.add_argument('--model', default='deepseek-chat')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=512)
    args = parser.parse_args()
    
    dataset_name = args.dataset_dir.strip(".").strip("\\").strip("/")
    print(dataset_name)
    prompt_path = os.path.join(args.dataset_dir, args.prompt_file)
    output_path = os.path.join(args.dataset_dir, args.output_file)
    
    if dataset_name == "ml-100k" or dataset_name == "ml-1m":
        inter_type = "观影"
        pref_list = "['Unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', " \
                    "'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']"
        example_msg =   f"样例如下\n" \
                        f"[输入：]用户0历史记录: ['Titanic'; 'Air Force One'; 'Apt Pupil'; 'Boogie Nights'; 'Good Will Hunting'; " \
                    "'Ulee's Gold'; 'L.A. Confidential'; 'Jerry Maguire'; 'In the Company of Men'; 'Full Monty, The'; 'English Patient, The'; " \
                        "'Scream'; 'Contact'; 'Mother'; 'Rosewood'; 'Crash'; 'Evita'; 'Liar Liar'; 'Murder at 1600'; 'Devil's Own, The']\n" + \
                        f"[输出：]用户0{inter_type}偏好: ['Drama', 'Thriller', 'Romance', 'Comedy', 'Crime']"
    elif dataset_name == "Amazon_Books":
        inter_type = "读书"
        pref_list = "['Romance', 'Children's Books', 'Mystery, Thriller & Suspense', 'Science Fiction & Fantasy', " \
                    "'Literature & Fiction', 'History', 'Biographies & Memoirs', 'Teen & Young Adult', 'Business & Money', 'Self-help']"
        example_msg =   f"样例如下\n" \
                        f"[输入：]用户0历史记录: ['Runaway (Starlight Animal Rescue)'; 'Loving Eliza'; 'Tested by Fire: He Sought Revenge - He Found Life']\n" + \
                        f"[输出：]用户0{inter_type}偏好: ['Children's Books', 'Romance', 'Literature & Fiction']"
    elif dataset_name == "lfm1b-tracks":
        inter_type = "音乐"
        pref_list = "['African', 'Asian', 'Avant-Garde', 'Blues', 'Caribbean & Latin American', 'Classical', 'Country', 'Easy Listening', 'Electronic', 'Folk', 'Hip Hop', 'Jazz', 'Pop', 'Rhythm & Blues', 'Rock', 'Ska']"
        example_msg =   f"样例如下\n" \
                        f"[输入：]用户0历史记录: ['In The Spring Twilight'; 'A Fool in Love'; 'Long Ago And Far Away'; 'Hvid Jul'; 'Santa Claus Is Comin' to Town'; 'Easy to Love'; 'Time To Say Goodbye (Con Te Partirè)'; 'Un Angelo Disteso Al Sole'; 'Fugt I Fundamentet'; 'Some Enchanted Evening']\n" + \
                        f"[输出：]用户0{inter_type}偏好: ['Classical', 'Jazz', 'Pop', 'Easy Listening']"
    else:
        inter_type = "选课"
        pref_list = ""
        example_msg =   f"[输入：]用户0历史记录: []\n" + \
                        f"[输出：]用户0{inter_type}偏好: []"

    system_msg = f"您需要根据用户历史记录预测用户的{inter_type}偏好，请从偏好列表中挑选：{pref_list}，并严格遵从样例格式回答。"
    

    if not os.path.exists(prompt_path):
        print(f"Error: path {prompt_path} not found")
        exit(1)
    
    print(f"Processing {prompt_path}，output is saved in {output_path}")
    cnt = 0
    
    with open(prompt_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            cnt += 1
            prompt_text = line.strip()
            if not prompt_text:
                continue
            
            print(f"Processing prompt {cnt}: {prompt_text[:30]}...")
            
            response = process_prompt(system_msg, example_msg, prompt_text)
            
            f_out.write(f"# Prompt {cnt}\n")
            f_out.write(prompt_text + "\n")
            f_out.write("=" * 50 + "\n")
            f_out.write(response + "\n\n")
            
            if cnt <= 5:
                print(f"Prompt: {prompt_text}")
                print(f"Response: {response}")
                print("=" * 50)
            
            time.sleep(1)
    
    print(f"Finished processing {cnt} lines of prompts")