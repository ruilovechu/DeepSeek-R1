from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import torch
import warnings
import threading

# q7b_fastapi.py
# 运行： uvicorn q7b_fastapi:app --reload
# swagger 文档：http://127.0.0.1:8000/docs#/
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

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

STOP_TOKENS = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
    tokenizer.convert_tokens_to_ids("</think>")  # 强制停止思考！
]

# 测试对话

def build_prompt(user_input):
    messages = [
        {"role": "system", "content": "你是严谨准确的AI助手。直接给出最终答案，**禁止输出任何思考、推理、分析过程**，不要出现标签，不要分步解释，只给简洁结果。不确定就直接说我不知道，绝不编造。"},
        {"role": "user", "content": user_input}
    ]
    # 模型官方模板，必须用这个！
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def chat_inner(prompt: str):
    print("\n用户：", prompt)
    if prompt == 'quit':
        return -1

    prompt = build_prompt(prompt)

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

        # 限制候选词，防止乱讲
        top_k=50,

        # 概率
        top_p=0.8, 

        # 关闭创意
        # do_sample=False,
        # temperature=0.1,

        # 秒出结果，不废话
        do_sample=True,

        # 无创意
        #temperature=0.1,

        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,

        # 关闭思考等待模式
        use_cache=True, 

        # 强制立刻出字
        #min_new_tokens=1

        # 轻微防重复
        repetition_penalty=1.1,
    )

    # 3. 用线程启动生成（不卡界面）
    thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    str_builder = ""

    # 4. 逐字输出
    print("\n", end="", flush=True)
    for new_text in streamer:
        # print(new_text, end="", flush=True)
        str_builder = str_builder + new_text
    print()

    return str_builder

# 本地测试
# while chat() == 0:
#     pass

class chat_request(BaseModel):
    content: str
    request_id: str

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {
        "item_id": item_id
        , "query": q
    }

@app.post("/chat")
async def chat(request: chat_request):
    print("用户输入：" + request.content)
    
    # 调用内部处理逻辑，裸模型处理提问，并返回回答
    chat_res = chat_inner(request.content)
    
    return {
        "data": chat_res
    }