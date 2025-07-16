import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributions import Categorical
import numpy as np
import os

model_name = "F:\model\Qwen-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

SYSTEM_PROMPT = """
            ## OutputFormat
            - <think>....</think> [digital number]
            ## Constrain
            - between the 2 <think> .... </think> label, show users your reasoning process, which indicate how to predict the final digital number
            - after the </think> label is the final answer, which should be only one number , no other characters, so that i can easily extract the final digital answer
        """


def print_model_output(outputs, inputs):
    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
    print("output_ids: ", tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n"))
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)  # 151668是</think>标签
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print("thinking_content:", thinking_content)
    print("Assistant:", content)


def generate_cot_and_entropy(input_text, max_length=2048):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {"role": "user", "content": input_text}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    except Exception as e:
        print("Error applying chat template:", e)
        text = input_text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        # do_sample=True、temperature=0.85 和 top_p=0.9 启用了采样模式。在采样模式下，model.generate 的 scores 只保留了采样选择的 token 的 logits，其他未选择的 token 的 logits 被置为 -inf
        # 这次是数学问题，要求推理精准，不需要答案的多样性，所以不需要采样
        do_sample=False,
        num_beams=1,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id
    )
    print_model_output(outputs.sequences, inputs)
    logits = outputs.scores  # 每个token的logit；维度是[token_num, vocab_size]
    # token_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:].squeeze(0)  # 去掉批次维度
    token_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:].flatten()  # Ensure 1D: [seq_len]
    entropies = []
    for logit in logits:
        logit = torch.where(torch.isinf(logit), torch.finfo(logit.dtype).tiny, logit)
        probs = F.softmax(logit, dim=-1)
        entropy = Categorical(probs=probs).entropy()
        entropies.append(entropy.item())
    print("Token IDs:", token_ids.tolist())
    print("Entropies:", entropies)
    return token_ids, entropies, logits


def select_high_entropy_tokens(token_ids, entropies, top_percent=0.2):
    num_tokens = len(entropies)
    num_high_entropy = max(1, int(num_tokens * top_percent))
    entropies_tensor = torch.tensor(entropies, dtype=torch.float32)# 转成tensor格式
    _, sorted_indices = torch.sort(entropies_tensor, descending=True)
    high_entropy_indices = sorted_indices[:num_high_entropy].tolist()
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    mask[high_entropy_indices] = True
    return mask

def policy_gradient_update(model, token_ids, rewards, entropy_mask, logits, lr=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    log_probs = []
    for logit in logits:
        logit = torch.where(torch.isinf(logit), torch.finfo(logit.dtype).tiny, logit)
        probs = F.softmax(logit, dim=-1)
        log_prob = torch.log(probs + 1e-10)
        log_probs.append(log_prob)
    log_probs = torch.stack(log_probs)[:-1]  # 形状: [seq_len-1, batch_size, vocab_size]：每个token在vocab中的distribution
    # 调整 token_ids_shifted 维度顺序
    token_ids_shifted = token_ids[1:].unsqueeze(1).unsqueeze(2)  # [seq_len-1, 1, 1]
    token_ids_expanded = token_ids_shifted.expand(-1, 1, 1)  # [seq_len-1, 1, 1]：每个token在vocab中的index
    token_log_probs = torch.gather(log_probs, dim=2, index=token_ids_expanded).squeeze(-1)  # [seq_len-1, batch_size]：每个token的实际概率
    loss = -token_log_probs * rewards
    # loss = loss * entropy_mask.to(device).float()
    # 对齐 entropy_mask 维度
    loss = loss * entropy_mask[1:].to(device).float()

    # 截断 rewards 和 entropy_mask
    # rewards = rewards[1:].to(device).float()  # [seq_len-1]
    # entropy_mask = entropy_mask[1:].to(device).float()  # [seq_len-1]
    # loss = -(token_log_probs * rewards * entropy_mask).mean()

    loss = loss.mean()
    loss.backward()
    optimizer.step()
    return loss.item()

def compute_rewards(token_ids, true_answer):
    generated_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    predicted_answer = generated_text.strip().split()[-1]  # 最后一个token作为answer
    return torch.ones_like(token_ids, dtype=torch.float) if predicted_answer == true_answer else torch.zeros_like(
        token_ids, dtype=torch.float)


def evaluate_aime_benchmarks(model, tokenizer, dataset):
    correct = 0
    total = len(dataset)
    for problem, true_answer in dataset:
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                    {"role": "user", "content": problem}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=True)
        except:
            text = problem
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True
        )
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        predicted_answer = generated_text.strip().split()[-1]  # 最后一个token作为answer
        if predicted_answer == true_answer:
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def train_rlvr(input_texts, true_answers, epochs=8, max_length=2048):
    for epoch in range(epochs):
        total_loss = 0.0
        for input_text, true_answer in zip(input_texts, true_answers):
            token_ids, entropies, logits = generate_cot_and_entropy(input_text, max_length)
            entropy_mask = select_high_entropy_tokens(token_ids, entropies)
            # entropy_mask = select_high_entropy_tokens(token_ids[1:], entropies[1:])
            rewards = compute_rewards(token_ids, true_answer)
            loss = policy_gradient_update(model, token_ids, rewards, entropy_mask, logits)
            total_loss += loss
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(input_texts)}")
    aime_accuracy = evaluate_aime_benchmarks(model, tokenizer, dataset=list(zip(input_texts, true_answers)))
    print(f"AIME Benchmark Accuracy: {aime_accuracy}")

    # 保存模型
    output_dir = "E:/LLM/models/finetuned_qwen_0.6B"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    input_texts = ["Solve the equation 2x + 3 = 7", "What is the value of x in 3x - 5 = 10?"]
    true_answers = ["2", "5"]
    train_rlvr(input_texts, true_answers)
