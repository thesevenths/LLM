AI教母李飞飞团队前几个月发了篇paper，介绍了一种低成本训练reasoning model的办法s1，资料如下：

Artifacts
- Paper: https://arxiv.org/abs/2501.19393
- Model: https://hf.co/simplescaling/s1.1-32B (Old: https://hf.co/simplescaling/s1-32B)
- Data: https://hf.co/datasets/simplescaling/s1K-1.1 (Old: https://hf.co/datasets/simplescaling/s1K)
- s1-prob: https://hf.co/datasets/simplescaling/s1-prob
- s1-teasers: https://hf.co/datasets/simplescaling/s1-teasers
- Full 59K: https://hf.co/datasets/simplescaling/data_ablation_full59K

该方法号称只用了1k条高质量的数据，准确率等指标接近deepseek r1和chatgpt o1，粗略看了其训练方法，用的就是sft，连RL都没用上，直观感觉没啥特别突出的地方，唯一可以拿出来说的“微创新”就是预算强制（Budget Forcing）了：
- 在推理阶段，他们引入了一种“预算强制”技术，通过延长模型的“思考时间”（即强制更深入的推理），提高答案准确性。这种方法类似于让模型在回答前进行更多思考。

这个所谓的Budget Forcing又是怎么做大的了？其实很简单：
- 提前终止（Early Termination） ：当模型生成的答案达到一定质量标准后，自动停止思考过程。
- 延长思考时间（Wait Instruction） ：如果模型认为当前的答案还不够完善，可以通过追加一个特殊的指令如“Wait”，告诉系统需要继续深入思考

不过要注意：该测试使用的是Qwen 32B的模型，并不是从0开始的，属于是“站在巨人肩膀”的那种！