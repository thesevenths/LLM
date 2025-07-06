from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen\Qwen3-0.6B")
llm = LLM(model="Qwen\Qwen3-0.6B", gpu_memory_utilization=0.15)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=32768,
    skip_special_tokens=False
)

prompt = '9.11和9.8谁大？'
prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"


outputs = llm.generate(
    prompt,
    sampling_params
)
print(f'原始输出：{prompt}{outputs[0].outputs[0].text}')
print('+'*20)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=32768,
    stop='</think>',
    skip_special_tokens=False
)

outputs = llm.generate(
        prompt,
        sampling_params
    )
wait = 'Wait'  # 增加reasoning的内容，提升准确率
for i in range(1):
    prompt += outputs[0].outputs[0].text + wait

    outputs = llm.generate(
        prompt,
        sampling_params
    )

print(f'wait后的输出：{prompt}{outputs[0].outputs[0].text}')
print('+'*20)
prompt += outputs[0].outputs[0].text
stop_token_ids = tokenizer("<|im_end|>")["input_ids"]
sampling_params = SamplingParams(
    max_tokens=32768,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
    skip_special_tokens=False,
    temperature=0.0,
)
outputs = llm.generate(
    prompt,
    sampling_params=sampling_params,
)

print(f'最后的输出：{prompt}{outputs[0].outputs[0].text}')


