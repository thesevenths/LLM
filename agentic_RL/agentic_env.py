# agentic_env.py
"""
    定义env和相关数据结构
    Env 负责执行 Planner 的 Action，维护对话context和memory
    支持的 Action 有 THINK / TOOL / ANSWER
"""

import abc
from typing import List, Any, Dict
import re

class ActionType:
    THINK = "THINK"
    TOOL = "TOOL"
    ANSWER = "ANSWER"

class Action:
    """高层行动决策（由 Planner 输出）"""
    def __init__(self, action_type: str, content: str, generated_content: str = None):
        """
        action_type: THINK / TOOL / ANSWER
        content: 对于 THINK，content 可以是 prompt 片段；对于 TOOL，可是工具名或参数；对于 ANSWER，是 final answer prompt
        """
        self.action_type = action_type
        self.content = content
        self.generated_content = generated_content  # 新增属性

class Observation:
    """给 Planner / Executor / Verifier 的输入观察上下文"""
    def __init__(self, context: str, memory: Dict[str, Any]):
        self.context = context
        self.memory = memory  # 可存历史工具调用、思考片段等

class PlannerStep:
    def __init__(self, act_id: int, content_id: int, old_logp: float, context_input_ids: List[int]):
        self.act_id = act_id
        self.content_id = content_id
        self.old_logp = old_logp
        self.context_input_ids = context_input_ids  # context at that step 的 input_ids snapshot

class Trajectory:
    def __init__(self):
        self.planner_steps: List[PlannerStep] = []
        self.planner_actions = []  # 存储所有的action
        self.contexts = []         # 存储每一步的context
        self.tool_results = []     # 存储工具调用结果
        self.think_texts = []      # 存储思考文本
        self.generated_answers = [] # 存储模型生成的答案
        self.final_answer = ""     # 最终答案
        self.reward = 0.0         # 总奖励
        self.local_rewards = []    # 每一步的局部奖励

class BaseEnv(abc.ABC):
    @abc.abstractmethod
    def reset(self, question: str) -> Observation:
        pass

    @abc.abstractmethod
    def step(self, action: Action) -> (Observation, float, bool):
        """
        执行动作得到下一个 observation、局部 reward、是否结束
        """
        pass


class SearchEnv(BaseEnv):
    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.question = None
        self.context = ""
        self.memory = {}
        self.done = False

    def reset(self, question: str) -> Observation:
        self.question = question
        self.context = f"Question: {question}\n"
        self.memory = {}
        self.done = False
        return Observation(self.context, self.memory)

    def step(self, action: Action):
        """
        执行 Planner 给出的 action。
        返回 (obs, reward, done)
        """
        if self.done:
            raise RuntimeError("Env already done")

        if action.action_type == ActionType.THINK:
            # THINK action: 把 content 加入 context
            self.context += f"<think>{action.content}</think>\n"
            # no immediate reward, not done
            return Observation(self.context, self.memory), 0.0, False

        elif action.action_type == ActionType.TOOL:
            # call tool (search)
            query = action.content
            result = self.search_engine.search(query)
            # insert info
            self.context += f"<search>{query}</search>\n"
            self.context += f"<information>{result}</information>\n"
            # local reward: e.g. if result not "No information found"
            r = 1.0 if "No information found" not in result else -0.5
            self.memory.setdefault("tool_calls", []).append((query, result))
            return Observation(self.context, self.memory), r, False

        elif action.action_type == ActionType.ANSWER:
            # final answer
            answer = action.content
            self.context += f"<answer>{answer}</answer>\n"
            self.done = True
            # final reward: correctness check
            # ground_truth should be stored in memory or env
            gt = self.memory.get("ground_truth", "")
            # todo: sematic match by another LLM, not just exact charactor match
            if gt and answer.strip() == gt.strip(): 
                return Observation(self.context, self.memory), 2.0, True
            else:
                return Observation(self.context, self.memory), 0.0, True

        else:
            raise ValueError("Unknown action type")