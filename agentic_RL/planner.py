# planner.py
"""
    根据 Observation 决定下一个 Action
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from agentic_env import Action, ActionType, Observation

class Planner:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def decide(self, obs: Observation) -> Action:
        """
        根据上下文 obs.context 决定下一个 action
        这里用简单 prompt + sampling 方式（可换policy network / 微调版）
        """
        prompt = obs.context + "\nWhat do you do next? Choose THINK / TOOL / ANSWER.\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=20, do_sample=True)
        out = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        # 根据输出判断 action type
        out_lower = out.lower()
        if "tool:" in out_lower:
            # parse tool content
            # 格式假定 “tool: search <query>”
            m = out_lower.split("tool:")[1].strip()
            parts = m.split(" ", 1)
            if parts[0] == "search" and len(parts) > 1:
                return Action(ActionType.TOOL, parts[1].strip())
        elif "answer:" in out_lower:
            m = out_lower.split("answer:")[1].strip()
            return Action(ActionType.ANSWER, m)
        # 默认 THINK
        return Action(ActionType.THINK, out.strip())
