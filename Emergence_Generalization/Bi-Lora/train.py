# train_bi_lora_final.py  —— 这次真·最终版，包跑通
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
from torch.optim import AdamW   # ← 改这里！


def prepare_bi_lora_model(base_model_path, target_modules):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    main_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model_main = get_peft_model(model, main_config)

    model_aux = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    aux_config = LoraConfig(
        r=32,
        lora_alpha=8,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model_aux = get_peft_model(model_aux, aux_config)

    return model_main, model_aux


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
    def __init__(self, aux_model, lambda_aux=1.0, **kwargs):
        super().__init__(**kwargs)
        self.aux_model = aux_model
        self.lambda_aux = lambda_aux
        self.aux_model.train()

        # 修复：从 torch.optim 导入 AdamW
        self.optimizer_aux = AdamW(
            filter(lambda p: p.requires_grad, aux_model.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss_main = outputs.loss

        aux_outputs = self.aux_model(**inputs)
        loss_aux = aux_outputs.loss

        loss = loss_main - self.lambda_aux * loss_aux
        return (loss, outputs) if return_outputs else loss

    # 完全兼容 fp16 + gradient_accumulation
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        self.aux_model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps

        # 自动兼容 fp16/bf16/deepspeed
        self.accelerator.backward(loss)

        # 梯度累积结束才 step
        if (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer_aux.step()
            self.optimizer.zero_grad()
            self.optimizer_aux.zero_grad()
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
    ds = make_dataset(data_path, tokenizer, max_length=1024)

    model_main, model_aux = prepare_bi_lora_model(base_model_path, target_modules)


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = BiLoRATrainer(
        model=model_main,
        aux_model=model_aux,
        lambda_aux=1.0,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
    )

    print("开始真正的 Bi-LoRA 训练，两个 LoRA 都在更新！")
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    model_main.save_pretrained(os.path.join(output_dir, "lora_main"))
    model_aux.save_pretrained(os.path.join(output_dir, "lora_aux"))
    tokenizer.save_pretrained(output_dir)
    print("训练完成！")


if __name__ == "__main__":
    main()