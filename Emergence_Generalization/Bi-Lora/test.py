from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import torch


base_model_path = r"F:\model\Qwen-0.6B"
data_path = r"F:\data\gsm8k"
output_dir = "./bi_lora_final"
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

main_config = LoraConfig(
    r=16,                   # 小模型建议 16~32
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, main_config)  # 先加默认配置
model.load_adapter("./bi_lora_final/lora_main", adapter_name="default")
# model.load_adapter("./bi_lora_final/lora_aux", adapter_name="aux")  # 推理时不用 aux
model.set_adapter("default")