* 纯文本LLM在pre-train阶段普遍使用auto regression方式，无需人工标注
* post-train阶段就要分情况了：
  * 传统的DPO/PPO需要人工标注数据：数学题需要题目、解题过程cot和最终答案
  * GRPO也需要人工标注数据：数学题只需要题目和答案，解题过程cot让LLM自行exploration
  * Test-Time Reinforcement Learning：数学题只需要题目，答案通过majority voting得到
  * Absolute Zero: Reinforced Self-play Reasoning with Zero Data：上述几种方法至少还需要使用人工生成或标注的data，这里的AZR连data都是LLM生成的：做数学题的时候题目、解题过程cot都是其他LLM生成的，完全不需要人工生成或标注数据了！
* 就现目前的技术手段而言，**train policy model肯定是需要数据的，否则哪来的信号让policy model做back propagattion和梯度下降了**？如果不用人工标注，那就只能额外用其他的model产生数据了！这里就有点“节外生枝”了：**policy model都还没训练好了，生成数据的model该怎么收敛了？**
* 同理：**reward如果用专门的model来提供，reward model又该怎么收敛了？ critical如果也用专门的model来判断，那么critical model应该怎么收敛了**？