import torch
from torch.utils.data import DataLoader
from model import Qwen3NextMoE
from data import get_dataloader
import deepspeed
from torch.optim import AdamW
from torch.cuda.amp import GradScaler

def train_model(config):
    model = Qwen3NextMoE(vocab_size=config['vocab_size'], hidden_size=config['hidden_size'])
    model.gradient_checkpointing_enable()  # Memory save

    # DeepSpeed init
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config_params=config['deepspeed_config']
    )

    dataloader = get_dataloader(config['batch_size'], config['block_size'])
    scaler = GradScaler()

    for step, batch in enumerate(dataloader):
        if step > config['max_steps']: break
        batch = batch.to(model_engine.device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = model_engine(inputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        model_engine.log({"loss": loss.item()}, step=step)
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    config = {
        'vocab_size': 50257,
        'hidden_size': 1024,
        'batch_size': 1,
        'block_size': 512,
        'max_steps': 10000,  # Adjust
        'deepspeed_config': 'ds_config.json'
    }
    train_model(config)