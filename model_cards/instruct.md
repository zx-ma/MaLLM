---
license: apache-2.0
language: en
datasets:
  - roneneldan/TinyStories
  - roneneldan/TinyStoriesInstruct
base_model: zhexMa/mallm-base
---

# mallm-instruct

mallm-base fine-tuned on TinyStories-Instruct to write stories from structured prompts.

- 10k steps, batch size 16, 512-token context; loss on story tokens only

## prompt format

```
<|user|>
Summary: Alice goes into the forest and finds a hidden lake.
Words: brave, lake, map
<|assistant|>
```

use any combination of `Summary`, `Words`, `Features` (e.g. `Dialogue`, `BadEnding`), and `Random sentence`, in any order. generation ends at `<|endoftext|>`.

## usage

```python
import json, sys
from huggingface_hub import snapshot_download
from safetensors.torch import load_model
from tokenizers import Tokenizer

path = snapshot_download("zhexMa/mallm-instruct")
sys.path.insert(0, path)
from model import Mallm, MallmConfig

model = Mallm(MallmConfig(**json.load(open(f"{path}/config.json"))))
load_model(model, f"{path}/model.safetensors")
tok = Tokenizer.from_file(f"{path}/tokenizer.json")
```

for generation, see `generate.py` in the [code repository](https://github.com/zx-ma/MaLLM).

## limitations

suited to english children's stories. expects the fields above, not free-form instructions. may miss required words.
