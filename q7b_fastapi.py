# q7b_fastapi.py
# 运行： uvicorn q7b_fastapi:app --reload
# swagger 文档：http://127.0.0.1:8000/docs#/

from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "query": q}