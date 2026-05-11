## 下载模型
```
DeepSeek-R1-Distill-Qwen-7B
```

## 检查 CUDA 最高版本
```shell
nvidia-smi
# 这里显示 CUDA Version: 13.1 
```

## 先手动创建好目录
```
D:\pip_packages
```

## 【废弃】下载 torch
```shell
pip download torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 --no-deps -d D:\pip_packages
```

## 【废弃】下载其它包
```shell

# 8bit 量化核心
pip download bitsandbytes==0.43.3 --no-deps -d D:\pip_packages

# GPU 加速
pip download accelerate==0.33.0 --no-deps -d D:\pip_packages

# 加载模型的核心库
pip download transformers==4.43.3 --no-deps -d D:\pip_packages

# 分词器依赖
pip download tokenizers==0.19.1 --no-deps -d D:\pip_packages

# torch 依赖 sympy
pip download sympy==1.12 --no-deps -d D:\pip_packages
pip download mpmath==0.19 --no-deps -d D:\pip_packages
pip download filelock --no-deps -d D:\pip_packages
pip download jinja2 --no-deps -d D:\pip_packages
pip download MarkupSafe==2.0 --no-deps -d D:\pip_packages
pip download networkx --no-deps -d D:\pip_packages
pip download fsspec --no-deps -d D:\pip_packages
pip download typing-extensions==4.8.0 --no-deps -d D:\pip_packages
pip download transformers==4.43.3 --no-deps -d D:\pip_packages
pip download numpy==1.26.2 --no-deps -d D:\pip_packages
pip download packaging==20.0 --no-deps -d D:\pip_packages
pip download six --no-deps -d D:\pip_packages
pip download huggingface-hub==0.23.2 --no-deps -d D:\pip_packages
```

## 注意
> 只有 torch / torchvision / torchaudio 需要加 CUDA 地址，其他全部默认源就行。

## 【废弃】安装
``` shell
pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\mpmath-0.19.tar.gz"
pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\sympy-1.12-py3-none-any.whl"
pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\filelock-3.29.0-py3-none-any.whl"
pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\MarkupSafe-2.0.0.tar.gz"
pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\jinja2-3.1.6-py3-none-any.whl"
pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\networkx-3.4.2-py3-none-any.whl"
pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\fsspec-2026.4.0-py3-none-any.whl"
pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\typing_extensions-4.8.0-py3-none-any.whl"
pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\torch-2.4.0+cu121-cp310-cp310-win_amd64.whl"
pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\numpy-1.26.2-cp310-cp310-win_amd64.whl"
pip install --no-index --find-links="D:\pip_packages" "D:\pip_packages\six-1.17.0-py2.py3-none-any.whl"
```

## 准备 requirements.txt
```
torch==2.4.1
torchvision==0.19.1
torchaudio==2.4.1
transformers==4.44.1
tokenizers==0.19.1
accelerate==0.34.0
bitsandbytes==0.44.0
huggingface-hub==0.24.6
numpy
packaging
pyyaml
requests
tqdm
protobuf
sympy
mpmath
filelock
jinja2
MarkupSafe
networkx
fsspec
typing-extensions
six
```

## 执行：

```shell
# pip download -r requirements.txt --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple -d D:\pip_packages
pip download -r requirements.txt --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple -d D:\pip_packages
```

## 进虚拟环境后执行
```shell
pip install --no-index --find-links=D:\pip_packages -r requirements.txt 
```

# FastAPI

## 向 requirements.txt 最后追加
```
fastapi>=0.110.0
uvicorn>=0.27.0
```

## 执行：

```shell
pip download -r requirements.txt --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple -d D:\pip_packages
pip install --no-index --find-links=D:\pip_packages -r requirements.txt 
```