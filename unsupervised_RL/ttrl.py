import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType
from collections import Counter

SYSTEM_PROMPT = """
response in the following format：
<think>
...
</think>
<answer>
...
</answer>
"""


def process_data(data):
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question_zh-cn']}
        ],
        'answer': x['answer_only']
    })
    return data


def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125

    if text.count("</think>\n") == 1:
        reward += 0.125

    if text.count("<answer>\n") == 1:
        reward += 0.125

    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward


# 生成答案是否正确的奖励
def correctness_reward(prompts, completions, **kwargs):
    # 从每个完成结果中提取答案
    responses = [extract_answer(completion[0]['content']) for completion in completions]

    # 统计提取的答案，找出出现次数最多的答案
    counter = Counter(responses)
    most_common = counter.most_common(1)  # 获取出现次数最多的答案
    answer = most_common[0][0]  # 最常见的答案

    # 构造与完成结果数量相同的答案列表
    answers = [answer] * len(completions)

    # 打印详细信息
    print('-' * 20)
    print(f"\n问题:\n{prompts[0][-1]['content']}")
    print(f"\n答案:\n{answers[0]}")
    print(f"\nLLM:\n{completions[0][0]['content']}")
    print(f"\nExtracted:\n{responses[0]}")

    # 返回奖励值：如果提取的答案与共识答案一致，则奖励为 1；否则为 -1
    return [1 if response == answer else -1 for response, answer in zip(responses, answers)]


def digit_reward(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]


def hard_format_reward(completions, **kwargs):
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


def mark_reward(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    return [mark_num(response) for response in responses]


if __name__ == '__main__':
    model_name = "Qwen\Qwen3-0.6B"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds = load_dataset('data\gsm8k')
    data = process_data(ds['train'])

    output_dir = "output"

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=200,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        report_to="tensorboard"
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            mark_reward,
            soft_format_reward,
            hard_format_reward,
            digit_reward,
            correctness_reward
        ],
        args=training_args,
        train_dataset=data,

    )
    trainer.train()
    trainer.save_model(output_dir)
