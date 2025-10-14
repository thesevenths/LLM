通过让 LLM 生成特殊标签（`<think>`, `<search>`, `<information>`, `<answer>`），并用 RL（PPO 变体）优化生成policy，实现了 Agentic RL 的核心：agent通过多次试错、尝试，学习在合适时机进行think、tool call和最终的answer生成。reward fun 驱动模型学习结构化行为和正确性，PPO 确保policy逐步改进，最终**让model“自主”决定的action sequence。 Agentic RL 的本质：结合 LLM 和 RL，让agent自主学会plan和tool call的能力。**

qwen3 0.6B的指令遵从能力还行：

```
1. 输入文本:
Question: system: 
            - you are an agentic assistance, you need to answer the question based on your knowledge and tools;
            - you know exactly how to plan 、execute and when to finish;
                - for simple questions, you can answer directly;
                - for complex questions, you can think \ tool \ answer in multiple steps;
            - your answer should be strictly in the following json format with keys:
                {'action': '...', 'content': [{'think': '...', 'tool': '...', 'answer': '...'}]};
            - finally, provide a concise and accurate answer.

user: What is Python?

```json
{
  "action": "answer",
  'action_content': {
    "content": [
      {
        "think": "",
        'tools': [],
        # the content here is the answer
        answer: "Python is a high-level, interpreted, general-purpose programming language that was first developed in 1989. It is widely used in various fields such as science, mathematics, data analysis, web development, and more."
      }
    ]
  }
}
```

```

```
