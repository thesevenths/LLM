# generator.py
"""
    生成final answer
"""

from agentic_env import Observation, Action, ActionType

class Generator:
    def __init__(self):
        pass

    def produce_answer(self, obs: Observation) -> Action:
        # 从 obs.context 提取 <answer>…</answer> 部分
        text = obs.context
        import re
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if m:
            return Action(ActionType.ANSWER, m.group(1).strip())
        # 否则 fallback
        # 这里可以扩展更复杂的生成模型
        fallback = "I am not sure."
        return Action(ActionType.ANSWER, fallback)
