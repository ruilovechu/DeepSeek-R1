from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import torch
import warnings
import threading

# 全局屏蔽警告
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# 模型路径
model_path = "models/DeepSeek-R1-Distill-Qwen-7B"

# 8bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("正在加载 7B 模型（GPU 8bit 量化）...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # torch_dtype=torch.bfloat16,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# 测试对话
prompt = input("输入你的问题：")
print("\n用户：", prompt)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 1. 创建流式生成器
streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,        # 跳过问题，只输出回答
    skip_special_tokens=True # 去掉特殊符号
)

# 2. 把 streamer 放进 generate
generate_kwargs = dict(
    **inputs,
    streamer=streamer,
    max_new_tokens=512,
    top_p=0.9,
    do_sample=False,
    temperature=0.1,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# 3. 用线程启动生成（不卡界面）
thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
thread.start()

# 4. 逐字输出
print("\nAI：", end="", flush=True)
for new_text in streamer:
    print(new_text, end="", flush=True)
print()

# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# print("\nAI 思考中...\n")
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=512,        
#         top_p=0.9,

#         # 有创意与无创意参数
#         # do_sample=True,
#         # temperature=0.7,
#         do_sample=False,
#         temperature=0.1,

#         pad_token_id=tokenizer.eos_token_id,
#         eos_token_id=tokenizer.eos_token_id
#     )

# # 只输出回答
# response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

# print("AI：", response)