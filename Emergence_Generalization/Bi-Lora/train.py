import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW  


def prepare_bi_lora_model(base_model_path, target_modules):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    main_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    aux_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 主 LoRA（默认名字就是 "default"）
    model = get_peft_model(base_model, main_config)

    # 辅助 LoRA
    model.add_adapter(adapter_name="aux", peft_config=aux_config)

    #  PEFT 启用多 adapter 融合
    from peft import PeftConfig
    model.peft_config["aux"] = aux_config  # 确保 aux 真的在 peft_config 里

    return model


def make_dataset(data_path, tokenizer, max_length=512):
    ds = load_dataset("parquet", data_files={
        "train": os.path.join(data_path, "train-00000-of-00001.parquet"),
        "validation": os.path.join(data_path, "test-00000-of-00001.parquet"),
    })

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(ex):
        prompt = "问题: " + ex["question"] + "\n回答: "
        full = prompt + ex["answer"] + tokenizer.eos_token

        tokenized = tokenizer(
            full,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        input_ids = tokenized["input_ids"]
        labels = input_ids[:]

        # mask prompt
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names, num_proc=4)
    return ds


class BiLoRATrainer(Trainer):
    def __init__(self, lambda_aux=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_aux = lambda_aux

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # 临时只用主 LoRA
    #     with model.set_adapter("default"):
    #         outputs_main = model(**inputs)
    #         loss_main = outputs_main.loss

    #     # 临时只用辅助 LoRA
    #     with model.set_adapter("aux"):
    #         outputs_aux = model(**inputs)
    #         loss_aux = outputs_aux.loss

    #     # Bi-LoRA 核心
    #     loss = loss_main - self.lambda_aux * loss_aux

    #     # 返回时可以附带主输出（用于日志、eval）
    #     return (loss, outputs_main) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False):
        # 直接指定 active_adapter，最保险
        loss_main = model(**inputs, active_adapter="default").loss
        loss_aux  = model(**inputs, active_adapter="aux").loss
        # loss_aux稍微变动一点，如果loss变化很大，说明当前landscape很sharp，一点都不flat
        loss = loss_main - self.lambda_aux * loss_aux
        return (loss, None) if return_outputs else loss


    # 兼容新版 Trainer 的第四个参数
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        if (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return loss.detach()

def main():
    base_model_path = r"F:\model\Qwen-0.6B"
    data_path = r"F:\data\gsm8k"
    output_dir = "./bi_lora_final"

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]  # 更强但更吃显存

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ds = make_dataset(data_path, tokenizer, max_length=512) # 相比1024省显存

    # 关键：只调用一次 prepare_bi_lora_model，返回一个 model
    model = prepare_bi_lora_model(base_model_path, target_modules)


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,        # 可以开大一点
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=3e-4,
        fp16=True,
        weight_decay=1e-4,                     
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,             # Windows 必关
    )

    trainer = BiLoRATrainer(
        model=model,               # 只传一个 base model！
        lambda_aux=1.0,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("开始真正的 Bi-LoRA 训练，两个 LoRA 都在更新！")
    trainer.train()

    # 保存时分别保存两个 adapter
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, "lora_main"), adapter_name="default")
    model.save_pretrained(os.path.join(output_dir, "lora_aux"),  adapter_name="aux")
    tokenizer.save_pretrained(output_dir)
    print("训练完成！两个 adapter 已分别保存")


if __name__ == "__main__":
    main()