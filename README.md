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
### eval qwen3-sft-too_call-chat

- chat

```
prompt:什么是机器学习？请简单解释。
response: 机器学习是一种人工智能技术，它使计算机能够从数据中学习并改进其性能，而无需进行明确的编程。通过分析大量数据，机器学习算法可以识别模式、趋势和关系，并使用这些信息来做出预测或决策。这种技术在许多领域都有应用，包括自然语言处理、计算机视觉、推荐系统和金融分析等。

prompt: 请以 JSON 格式列出中国四大发明
response: 的名称和发明者。 以下是中国四大发明的名称和发明者列表：
{
  "inventions": [
    {
      "name": "造纸术",
      "inventor": "蔡伦"
    },
    {
      "name": "印刷术",
      "inventor": "毕昇"
    },
    {
      "name": "火药",
      "inventor": "唐高宗"
    },
    {
      "name": "指南针",
      "inventor": "张衡"
    }
  ]
}

prompt:小明有 15 个苹果，给了小红 3 个，又买了 7 个，现在有多少个？
response: 小明有 15 个苹果，给了小红 3 个，那么小明还剩下 15 - 3 = 12 个苹果。
小明又买了 7 个苹果，那么现在小明一共有 12 + 7 = 19 个苹果。
所以，小明现在有 19 个苹果。

prompt:Translate the following to English: 人工智能正在改变世界
response:，它将带来更多的机会和挑战。 人工智能 is changing the world, bringing more opportunities and challenges.
```

- tool
```
prompt:<tools>[{"name": "get_paper", "description": .......}]</tools>Which papers citing 'PHILOSOPHY OF EDUCATION' were published in 2022?
Response: Based on the available API, I can't directly retrieve papers citing 'PHILOSOPHY OF EDUCATION' published in 2022. The API does not provide a direct search for papers citing a specific paper. However, I can help you with other related searches or provide information about the 'PHILOSOPHY OF EDUCATION' paper itself if needed. Let me know how I can assist you further.
```

# GRPO

>从 qwen3-sft-tool_call-chat 续训