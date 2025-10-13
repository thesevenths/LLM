# run_agent.py

from trainer import Trainer
from agentic_env import SearchEnv
from transformers import AutoTokenizer
import argparse

# 一个简易的 “知识库 / 搜索引擎” 实现；这里只是简单用字典模拟，做字符串匹配。严谨做法需要用embedding+向量数据库做sematic语义检索
class SearchEngine:
    def __init__(self):
        self.kb = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "machine learning": "Machine learning is a subset of AI that enables computers to learn from experience.",
            "neural networks": "Neural networks are computing systems inspired by biological neural networks.",
            "deep learning": "Deep learning is a subset of machine learning using artificial neural networks.",
            "transformer": "Transformers are neural network architectures using self-attention mechanisms.",
            "reinforcement learning": "Reinforcement learning involves agents learning through environment interaction."
        }
    def search(self, query: str):
        q = query.lower().strip()
        for k, v in self.kb.items():
            if k in q:
                return v
        return f"No information found for: {query}"

def main():
    se = SearchEngine()
    trainer = Trainer(se, model_name="F:\\model\\Qwen-0.6B", lr=1e-5)  # F:\\model\\Qwen3-0.6B

    # 准备训练数据
    queries = ["What is Python?", "Explain machine learning", "What is deep learning?"]
    truths = ["Python is a programming language", "Machine learning is AI subset", "Deep learning uses neural networks"]

    print("Training...")
    trainer.train(queries, truths, epochs=5)

    print("Testing...")
    trainer.test("What is neural networks?")

if __name__ == "__main__":
    main()
