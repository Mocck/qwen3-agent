# SFT

### original datasets

```python
from datasets import load_dataset

tool = load_dataset("allenai/Dolci-Instruct-SFT-Tool-Use", split="train")
chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
```
### pre-process datasets

>每次训练前，从原始数据集的一部分子集中，随机采样一定数量样本， 按照1:1比例混合两种数据，然后使用 process_message 函数对数据进行处理。

### SFT results

1) tool+chat:
   
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base")
model = PeftModel.from_pretrained(base_model, "moccck/qwen3-sft-tool-chat")
```

2) tool_call+chat:
   
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base")
model = PeftModel.from_pretrained(base_model, "moccck/qwen3-sft-tool_call-chat")
```

- qwen3-sft-too-chat: Dolci-Instruct-SFT-Tool-Use数据包含完整对话过程；ultrachat_200k数据包含完整对话过程。

- qwen3-sft-too_call-chat: Dolci-Instruct-SFT-Tool-Use数据只包含工具调用部分；ultrachat_200k数据包含完整对话过程。

---
# GRPO

>从 qwen3-sft-tool_call-chat 续训