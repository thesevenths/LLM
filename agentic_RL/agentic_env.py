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
    def __init__(self, action_type: str, content: str = ""):
        """
        action_type: THINK / TOOL / ANSWER
        content: 对于 THINK，content 可以是 prompt 片段；对于 TOOL，可是工具名或参数；对于 ANSWER，是 final answer prompt
        """
        self.action_type = action_type
        self.content = content

class Observation:
    """给 Planner / Executor / Verifier 的输入观察上下文"""
    def __init__(self, context: str, memory: Dict[str, Any]):
        self.context = context
        self.memory = memory  # 可存历史工具调用、思考片段等

class Trajectory:
    """存储一次 rollout 的数据"""
    def __init__(self):
        self.planner_actions: List[Action] = []
        self.tool_results: List[Any] = []  # 每次 TOOL 对应的结果
        self.think_texts: List[str] = []  # THINK 阶段的思考文本
        self.final_answer: str = ""
        self.reward: float = 0.0
        self.local_rewards: List[float] = []  # 各 action / step 的局部 reward
        self.contexts: List[str] = []  # 每步 context（可用于策略重演）

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
            if gt and answer.strip() == gt.strip():
                return Observation(self.context, self.memory), 2.0, True
            else:
                return Observation(self.context, self.memory), 0.0, True

        else:
            raise ValueError("Unknown action type")