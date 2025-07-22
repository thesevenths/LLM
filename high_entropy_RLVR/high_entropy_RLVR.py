import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os

model_name = "E:\model\Qwen-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16) # float16半精度节约显存
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 启用模型参数的梯度跟踪
for param in model.parameters():
    param.requires_grad = True

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

def generate_cot_and_entropy(input_text, max_length):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {"role": "user", "content": input_text}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    except Exception as e:
        print("Error applying chat template:", e)
        text = input_text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None).to(device) if inputs.get("attention_mask") is not None else None

    # 手动生成以保留计算图
    model.eval()
    with torch.enable_grad(), autocast():
        generated_ids = input_ids.clone()
        logits_list = []
        max_new_tokens = max_length - input_ids.shape[1]
        for _ in range(max_new_tokens):
            # 启用 Flash Attention（若可用）和 memory‑efficient attention，减少注意力层的内存开销
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
                outputs = model(generated_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            logits = torch.where(torch.isinf(logits), torch.finfo(logits.dtype).tiny, logits)
            probs = F.softmax(logits, dim=-1)
            if model.training:
                logits_list.append(logits.detach())
            else:
                logits_list.append(logits)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            del outputs, probs
            torch.cuda.empty_cache()
            if next_token.item() == tokenizer.eos_token_id:
                break

        # 在生成序列后一次性计算熵
        entropies = []
        for logits in logits_list:
            logits = torch.where(torch.isinf(logits), torch.finfo(logits.dtype).tiny, logits)
            probs = F.softmax(logits, dim=-1)
            entropy = Categorical(probs=probs).entropy()
            entropies.append(entropy.item())

    print_model_output(generated_ids, inputs)
    token_ids = generated_ids[0, input_ids.shape[1]:].flatten()  # Ensure 1D: [seq_len]
    print("Token IDs:", token_ids.tolist())
    print("Entropies:", entropies)
    return token_ids, entropies, logits_list

def select_high_entropy_tokens(entropies, top_percent=0.2):
    num_tokens = len(entropies)
    num_high_entropy = max(1, int(num_tokens * top_percent))
    entropies_tensor = torch.tensor(entropies, dtype=torch.float32)  # 转成tensor格式
    _, sorted_indices = torch.sort(entropies_tensor, descending=True)
    high_entropy_indices = sorted_indices[:num_high_entropy].tolist()
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    mask[high_entropy_indices] = True
    return mask

def policy_gradient_update(model, token_ids, rewards, entropy_mask, logits, lr=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    scaler = GradScaler()
    with autocast():
        log_probs = []
        for logit in logits:
            logit = torch.where(torch.isinf(logit), torch.finfo(logits.dtype).tiny, logit)
            probs = F.softmax(logit, dim=-1)
            log_prob = torch.log(probs + 1e-10)
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs)  # 形状: [seq_len, batch_size, vocab_size]：每个token在vocab中的distribution
        # 调整 token_ids_shifted 维度顺序
        token_ids_shifted = token_ids[1:].unsqueeze(0).unsqueeze(-1)  # [seq_len-1, 1, 1]
        token_log_probs = torch.gather(log_probs, dim=2, index=token_ids_shifted).squeeze(-1)  # [seq_len-1, 1]：从distribution中提取每个token的实际概率
        entropy_mask_reshaped = entropy_mask[1:].to(device).float().unsqueeze(-1)  # [seq_len-1, 1]
        loss = -token_log_probs * rewards[1:].unsqueeze(-1) * entropy_mask_reshaped  # [seq_len-1, 1]
        loss = loss.mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()

def compute_rewards(token_ids, true_answer):
    generated_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    predicted_answer = generated_text.strip().split()[-1]  # 最后一个token作为answer
    return torch.ones_like(token_ids, dtype=torch.float) if predicted_answer == true_answer else torch.zeros_like(token_ids, dtype=torch.float)

def train_rlvr(input_texts, true_answers, epochs=8, max_length=512):
    for epoch in range(epochs):
        total_loss = 0.0
        for input_text, true_answer in zip(input_texts, true_answers):
            token_ids, entropies, logits = generate_cot_and_entropy(input_text, max_length)
            entropy_mask = select_high_entropy_tokens(entropies)
            rewards = compute_rewards(token_ids, true_answer)
            loss = policy_gradient_update(model, token_ids, rewards, entropy_mask, logits)
            total_loss += loss
            torch.cuda.empty_cache()
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(input_texts)}")
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