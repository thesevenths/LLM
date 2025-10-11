# verifier.py
"""
    校验tool call的结果，给局部 reward
"""

class Verifier:
    def __init__(self):
        pass

    def verify_tool(self, query: str, result: str) -> float:
        """校验tool call结果，给局部 reward"""
        if "No information found" in result:
            return -1.0
        # 简单相关性检查：query 的关键词是否出现在 result
        if query.lower() in result.lower():
            return 0.5
        return 0.0

    def verify_think(self, think_text: str) -> float:
        """校验 think 内容合理性（可选）"""
        # 简单惩罚过长 / 空思考
        if not think_text or len(think_text) < 3:
            return -0.2
        return 0.0
