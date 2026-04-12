# pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\torch-2.1.0+cu118-cp310-cp310-win_amd64.whl"
# pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\transformers-4.38.2-py3-none-any.whl"   # 会把 numpy 2.2.6 安装上，这个版本是不兼容的
# pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\accelerate-0.24.0-py3-none-any.whl"
# pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\sentencepiece-0.1.99-cp310-cp310-win_amd64.whl"
# pip uninstall numpy -y
# pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\numpy-1.24.3-cp310-cp310-win_amd64.whl"

# TextIteratorStreamer： 流式输出
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer  
import torch

# 线程
from threading import Thread

def get_clean_response(origin_response):
    """
    摘要：
        清洗模型生成的原始输出，提取助手的第一句回答

    参数:
        origin_response: 模型生成的完整对话文本，包含 ### 用户、### 助手 等标记

    返回:
        清洗后的纯回答文本
    """
    # 1. 取出 ### 助手 后面的所有内容
    if "### 助手" in origin_response:
        origin_response = origin_response.split("### 助手")[1].strip()

        # 2. 只取 第一句话 （遇到换行/句号/### 就截断）
        # 按换行切割，取第一行
        origin_response = origin_response.splitlines()[0].strip()

    # 3. 清理多余内容
    origin_response = origin_response.strip()
    return origin_response

def get_clean_response2(origin_response):
    """
    摘要：
        清洗模型生成的原始输出，提取助手的第一句回答
    参数:
        origin_response: 模型生成的完整文本
    返回:
        干净的回答
    """
    # 1. 找到 ### 助手 后面的内容
    if "### 助手" in origin_response:
        origin_response = origin_response.split("### 助手")[1].strip()
    else:
        origin_response = origin_response.strip()

    # 2. 只要遇到 ### 就截断（后面全是垃圾）
    if "###" in origin_response:
        origin_response = origin_response.split("###")[0].strip()

    # 3. 去掉 </think> 这种奇怪标签
    origin_response = origin_response.replace("</think>", "").strip()

    return origin_response

def get_clean_response3(text):
    # 1. 先切  后面
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # 2. 再切掉后面可能出现的 ### 垃圾
    if "###" in text:
        text = text.split("###")[0].strip()

    return text.strip()


# 模型路径（改成你的实际路径）
model_path = "./models/DeepSeek-R1-Distill-Qwen-1.5B"

print("=" * 50)
print("✓ 开始加载模型...")
print(f"✓ 路径: {model_path}")

# 加载分词器
print("✓ 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型（GTX 1060 优化）
print("✓ 加载模型（使用 float16）...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print(f"✓ 模型已加载到: {model.device}")
print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 50)

while True:

    # 原始输入
    user_input = input("\n请输入提问内容（输入quit退出对话）：")

    if user_input == 'quit':
        break

    # # prompt
    # prompt = f"""以下是参考知识，请根据知识回答问题，不要编造。

    # 以下是用户与AI助手的对话。
    # 请你作为AI助手，直接、简洁地回答用户的问题，不要输出多余内容，不要重复格式。

    # ### 用户
    # {user_input}

    # ### 助手
    # """

    messages = [
        {"role": "system", "content": "你是一个专业智能助手，回答简洁、准确、不编造。"},
        {"role": "user", "content": user_input}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 流式输出器
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,  # 只输出回答，不输出问题
        skip_special_tokens=True
    )

    # generate 参数
    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.5,
        "top_p": 0.9,
        "repetition_penalty": 1.2, # 重复惩罚
        "no_repeat_ngram_size": 2, # 禁止重复 2-gram
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    # 后台线程生成
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # 实时逐字显示
    print("\n" + "*" * 50)
    response_text = ""
    for new_text in streamer:
        response_text += new_text.lstrip()
        print(new_text, end="", flush=True)
    print("\n" + "*" * 50)        

    # 经过清洗的完整回答
    print("\n" + "*" * 50)
    print(get_clean_response3(response_text), end="\n")
    print("\n" + "*" * 50)


    # # 解码完整输出
    # full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # # 提取最终内容
    # response = get_clean_response(full_output)
    # print(f"\n原始输出: {full_output}")
    # print("\n")
    # print("*" * 50)
    # print(f"\n输出: {response}")
    # print("*" * 50)
    # print("\n")