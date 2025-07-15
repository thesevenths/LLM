import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributions import Categorical
import numpy as np

# Load model and tokenizer
model_name = "E:\model\Qwen-0.6B"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def print_model_output(outputs, inputs):
    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
    print("output_ids:    ", tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n"))
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)  # 151668是</think>标签，从这里截断
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print("thinking_content:", thinking_content)
    print("Assistant:", content)


def generate_cot_and_entropy(input_text, max_length=2048):
    messages = [{"role": "user", "content": input_text}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
    except Exception as e:
        print("Error applying chat template:", e)
        text = input_text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.85,
        top_p=0.9,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id
    )
    print_model_output(outputs.sequences, inputs)
    logits = outputs.scores
    token_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
    entropies = []
    for logit in logits:
        probs = F.softmax(logit, dim=-1)
        entropy = Categorical(probs=probs).entropy()
        entropies.append(entropy.item())
    return token_ids, entropies


def select_high_entropy_tokens(token_ids, entropies, top_percent=0.2):
    num_tokens = len(entropies)
    num_high_entropy = max(1, int(num_tokens * top_percent))
    sorted_indices = np.argsort(entropies)[::-1]
    high_entropy_indices = sorted_indices[:num_high_entropy]
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    mask[high_entropy_indices] = True
    return mask


def policy_gradient_update(model, token_ids, rewards, entropy_mask, lr=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    inputs = token_ids.unsqueeze(0).to(device)
    outputs = model(inputs, labels=inputs)
    logits = outputs.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=2, index=inputs[:, 1:].unsqueeze(-1)).squeeze(-1)
    loss = -token_log_probs * rewards
    loss = loss * entropy_mask.to(device).float()
    loss = loss.mean()
    loss.backward()
    optimizer.step()
    return loss.item()


def compute_rewards(token_ids, true_answer):
    generated_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    predicted_answer = generated_text.strip().split()[-1]
    return torch.ones_like(token_ids, dtype=torch.float) if predicted_answer == true_answer else torch.zeros_like(
        token_ids, dtype=torch.float)


def evaluate_aime_benchmarks(model, tokenizer, dataset):
    correct = 0
    total = len(dataset)
    for problem, true_answer in dataset:
        messages = [{"role": "user", "content": problem}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=True)
        except:
            text = problem
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True
        )
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        predicted_answer = generated_text.strip().split()[-1]
        if predicted_answer == true_answer:
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def train_rlvr(input_texts, true_answers, epochs=8, max_length=2048):
    for epoch in range(epochs):
        total_loss = 0.0
        for input_text, true_answer in zip(input_texts, true_answers):
            token_ids, entropies = generate_cot_and_entropy(input_text, max_length)
            entropy_mask = select_high_entropy_tokens(token_ids, entropies)
            rewards = compute_rewards(token_ids, true_answer)
            loss = policy_gradient_update(model, token_ids, rewards, entropy_mask)
            total_loss += loss
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(input_texts)}")
    aime_accuracy = evaluate_aime_benchmarks(model, tokenizer, dataset=list(zip(input_texts, true_answers)))
    print(f"AIME Benchmark Accuracy: {aime_accuracy}")


if __name__ == "__main__":
    input_texts = ["Solve the equation 2x + 3 = 7", "What is the value of x in 3x - 5 = 10?"]
    true_answers = ["2", "5"]
    train_rlvr(input_texts, true_answers)
