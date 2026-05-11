from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch

EMBED_MODEL_PATH = "./models/bge-base-zh-v1.5"

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 加载向量化模型
emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
emb_model = AutoModel.from_pretrained(EMBED_MODEL_PATH)
emb_model.eval()

app = FastAPI(title="DeepSeek 7B + bge-base 本地知识库API")

class TextInput(BaseModel):
    text: str

@app.post("/vector")
def get_vector(data: TextInput):
    inputs = emb_tokenizer(
        [data.text],
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = emb_model(**inputs)

    # 生成标准向量
    vector = outputs.last_hidden_state[:, 0, :]
    vector = torch.nn.functional.normalize(vector, p=2, dim=1)

    return {
        "vector": vector[0].tolist(),
        "dim": 768  # bge-base 向量维度固定 768
    }