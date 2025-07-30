1、概述：  
* 纯文本LLM在pre-train阶段普遍使用auto regression方式，无需人工标注
* post-train阶段就要分情况了：
  * 传统的DPO/PPO需要人工标注数据：数学题需要题目、解题过程cot和最终答案
  * GRPO也需要人工标注数据：数学题只需要题目和答案，解题过程cot让LLM自行exploration
  * Test-Time Reinforcement Learning：数学题只需要题目，答案通过majority voting得到
  * Absolute Zero: Reinforced Self-play Reasoning with Zero Data：上述几种方法至少还需要使用人工生成或标注的data，这里的AZR连data都是LLM生成的：做数学题的时候题目、解题过程cot都是其他LLM生成的，完全不需要人工生成或标注数据了！
* 就现目前的技术手段而言，**train policy model肯定是需要数据的，否则哪来的信号让policy model做back propagattion和梯度下降了**？如果不用人工标注，那就只能额外用其他的model产生数据了！这里就有点“节外生枝”了：**policy model都还没训练好了，生成数据的model该怎么收敛了？**
* 同理：**reward如果用专门的model来提供，reward model又该怎么收敛了？ critical如果也用专门的model来判断，那么critical model应该怎么收敛了**？  

2、公式解读：  
* Q(s,a) = R(t) + gam*V(s+1) : action的value算法，**体现action本身的价值**; 纯NLP的LLM做RL，很难得到每个step的reward，所以这种计算Q的方式在LLM场景比较困难；  
* A(s,a) = Q(s,a) - V(s) = R(t) + gam*V(s+1) - V(s) : 当前action相比其他action，在s下带来的**增量价值**  
* 由以上第二个公式变形：Q(s,a) = A(s,a) + V(s), 如果能求出A(s,a)，岂不是就能得到Q(s,a)？  
