# executor.py
"""
    执行 Planner 的 Action，调用工具，返回结果给 Env
"""

class Executor:
    def __init__(self, search_engine):
        self.search_engine = search_engine

    def call_tool(self, tool_name: str, content: str):
        # 目前只实现 search 工具
        if tool_name == "search":
            return self.search_engine.search(content)
        else:
            raise NotImplementedError(f"Tool {tool_name} not implemented")
