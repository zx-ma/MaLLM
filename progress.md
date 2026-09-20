# mallm 项目进度笔记

> 这份笔记用于在新会话中恢复完整上下文。记录了项目目标、协作方式、已完成的工作、每个文件的当前状态、所有讨论过的知识点、踩过的坑，以及下一步计划。

---

## 1. 项目是什么

用户（Zhexuan Ma，zhexuanma@gmail.com）想**从零学习 LLM**，亲手把整条链路走一遍。硬件是 **8GB 显存的 RTX 5070 Laptop GPU（Blackwell, sm_120）**。

原始计划（用户最初粘贴的中文方案）：tokenizer → 在 TinyStories 上预训练 → 评估 → SFT → DPO → 可选的 VLM。

项目路径：`/home/mzx/projects/my/mallm`，git 仓库，当前分支 `master`（主分支名为 `main`）。

---

## 2. 协作方式（非常重要，必须遵守）

用户给过多次明确反馈，已存入 memory：

1. **教学式，不要给完整代码。** 原话："教会我要怎么做 bpe什么意思？ 不要告诉我完整的代码， 循循善诱，教会我"。用户会自己写代码，然后说"检查"/"check"，期望得到 review + 提示，而不是答案。
2. **不要替用户装包。** 原话："这种安装包之类的事情以后告诉我， 我自己来"。给命令，让用户自己跑。
3. **不要为了验证语义而跑代码。** 原话两次："没必要先跑一边吧， 直接看我写的就行"、"你直接回答我就行，没有必要运行代码来确认吧？"。例外（用户接受的）：检查磁盘上文件的真实状态（比如文件是否被 `wb` 清空、checkpoint 里有哪些键）。
4. **方案 review 保持高层次。** 原话："不用看太细, 具体怎么做我们还能调整的, 只是用于学习"。
5. **一次不要给太多信息。** 原话："简单易懂， 不要一次性给我过多的信息"、"我没有完全理解你说的， 一下子给我太多信息了"。用户经常要求"再详细易懂一点"、"慢点"、"举例子解释"。
6. **用户时间紧时会明确要求代写。** 例如"帮我把get补好吧"、"我想赶紧先写完去睡觉"。这时直接写代码是对的，写完简要解释。

全局 CLAUDE.md 规则：优先自解释代码；注释和 docstring 保持最少、小写英文、简洁。

Memory 文件位置：`~/.claude/projects/-home-mzx-projects-my-mallm/memory/`
- `MEMORY.md`（索引）
- `project-llm-from-scratch.md`
- `feedback-keep-reviews-high-level.md`
- `feedback-user-runs-installs.md`
- `feedback-answer-dont-verify.md`

---

## 3. 环境

- torch 2.11.0+cu128（pyproject.toml 里配了 pytorch-cu128 explicit index，Blackwell 需要）
- numpy 2.5.3、tokenizers 0.23.2、tqdm
- Python >= 3.13，uv 管理项目
- 运行方式：`uv run python xxx.py`；注意 shell 里没有裸 `python`，必须用 `uv run python` 或 `.venv/bin/python`
- 系统：Linux，KDE 桌面，zsh，终端 ghostty
- GPU 空闲时约 1.1GB 被桌面占用（Xorg 522MB、kwin 154MB、ghostty 125MB、plasmashell 86MB），可用约 7GB

---

## 4. 当前文件状态

### 4.1 `config.py`（完成）
```
TRAIN_TXT = "data/raw/TinyStoriesV2-GPT4-train.txt"
VALID_TXT = "data/raw/TinyStoriesV2-GPT4-valid.txt"
VOCAB_SIZE = 8192
SPECIALS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"]
TOKENIZER_PATH = f"tokenizer-{VOCAB_SIZE // 1024}k.json"
TRAIN_BIN = "data/processed/train.bin"
VAL_BIN = "data/processed/val.bin"
CKPT_PATH = "checkpoints/mallm.pt"
```

### 4.2 `bpe.py`（完成，学习用）
用户手写的 byte-level BPE：`get_pair_count`、`my_merge`、`train`、`build_vocab`、`encode`、`decode`。已验证 round trip 正确、压缩率 2.41 字节/token（在 500 词表上）、中文可通过字节回退还原。已知不修的局限：合并会跨越单词边界（出现 `b'. They '` 这种 token）、速度慢（100 万字符 244 次合并要 14.6 秒，推算 8192 次合并要几天）。

### 4.3 `train_tokenizer.py`（完成）
用 HF `tokenizers` 库：`Tokenizer(models.BPE())` + `pre_tokenizers.ByteLevel(add_prefix_space=False)` + `decoders.ByteLevel()` + `trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)`，在完整 2.1GB 训练文本上训练，保存为 `tokenizer-8k.json`。

词表构成：4 个特殊 token（id 0-3）+ 256 个字节 + 7932 次合并 = 8192。

### 4.4 `check_tokenizer.py`（只完成了 1/5）
目前只有 round-trip 断言。用户欠着的另外四项检查：词表大小、压缩率、4 个特殊 token 各自是单个 id、打印某句话的 `.tokens`。**优先级低，可以不做。**

### 4.5 `prepare.py`（完成并已运行）
最终结构：
- `batch2file(stories, tok, out_f)`：`tok.encode_batch` → 用 `ids.extend(e.ids)` 拼平 → `np.array(ids, dtype=np.uint16)` → `arr.tofile(out_f)` → `return len(arr)`
- `encode_file(tokenizer_path, text_path, bin_path)`：`Path(bin_path).parent.mkdir(parents=True, exist_ok=True)` → 逐行读、遇到 `line.strip() == "<|endoftext|>"` 就切出一篇故事 → 攒够 1000 篇调用一次 `batch2file` 并清空 buffer → 循环结束后 tail flush → `return n_docs, total`
- `if __name__ == "__main__":` 里调用

（注意：用户实际写的参数名有个拼写错误 `tokenzizer_path`，当时指出过，不确定是否已改。）

### 4.6 `model.py`（完成）
从上到下：
- `MLP(nn.Module)`：`__init__(self, d)`，`l1 = nn.Linear(d, 4*d)`、`l2 = nn.Linear(4*d, d)`、`ac1 = nn.GELU()`；forward 是 l1 → gelu → l2，**最后不再激活**
- `Attention(nn.Module)`：`__init__(self, d, n_head)`，存 `self.nh`，有 `q_project`/`k_project`/`v_project`/`fc` 四个 `nn.Linear(d, d)`
  - forward：`B, T, d = x.size()` → 三个投影 → 各自 `.view(B, T, self.nh, -1).transpose(1, 2)` 变成 `(B, H, T, hd)` → `scores = q @ k.transpose(-1, -2) / (q.size(-1) ** 0.5)` → `mask = torch.tril(torch.ones(T, T, device=x.device))` → `scores.masked_fill(mask == 0, float("-inf"))` → `torch.softmax(dim=-1)` → `alpha @ v` → `.transpose(1, 2).reshape(B, T, -1)` → `self.fc(out)`
- `Block(nn.Module)`：`__init__(self, d, n_head)` 有 `attn`、`ln1`、`mlp`、`ln2`（两个独立的 `nn.LayerNorm(d)`）；forward 是 pre-norm 残差：`x = x + self.attn(self.ln1(x))`、`x = x + self.mlp(self.ln2(x))`
- `MallmConfig`（dataclass）：字段顺序是 `vocab_size: int`（无默认值）、`d: int = 384`、`block_size: int = 256`、`n_head: int = 6`、`n_layer: int = 6`
- `Mallm(nn.Module)`：`__init__(self, config)`，存 `self.config = config`，还有一个冗余的 `self.vocab_size = config.vocab_size`（提过可以删，未删）
  - 组件：`tok_emb = nn.Embedding(config.vocab_size, config.d)`、`pos_emb = nn.Embedding(config.block_size, config.d)`、`blocks = nn.ModuleList([Block(config.d, config.n_head) for _ in range(config.n_layer)])`、`ln_final = nn.LayerNorm(config.d)`、`language_model_head = nn.Linear(config.d, config.vocab_size, bias=False)`
  - 权重共享：`self.tok_emb.weight = self.language_model_head.weight`（**这个方向是对的**，保留了 Linear 的小初始值，和 nanoGPT 一致）
  - `forward(self, index, target=None)`：`B, T = index.size()` → `tok_emb(index)` → `positions = torch.arange(T, device=index.device)` → `pos_emb(positions)` → 相加 → 过所有 block → `ln_final` → `language_model_head` → `loss = None`，若 target 非空则 `F.cross_entropy(logits.reshape(-1, self.vocab_size), target.reshape(-1))` → `return logits, loss`

### 4.7 `train.py`（可用的最小版本，正在升级）
- `get_batch(data, B, T, device)`：`starts = torch.randint(0, len(data) - T, (B,))` → 循环里 `row_x = data[start:start+T].astype(np.int64)`、`row_y = data[start+1:start+1+T].astype(np.int64)`，分别 append 到 `xs`/`ys` → 循环外 `torch.from_numpy(np.stack(xs))` → `return x.to(device), y.to(device)`
- `@torch.no_grad() def evaluate_loss(data, model, loop_time, batch_size, block_size, device)`：`model.eval()` → 循环 `loop_time` 次取 batch 前向收集 `loss.item()` → `model.train()` → 返回平均值
- `save_ckpt(mallm, ckpt_path)`：存 `{"model": mallm.state_dict(), "config": asdict(mallm.config)}`
- `__main__`：加载两个 memmap → `cfg = MallmConfig(vocab_size=VOCAB_SIZE)` → `batch_size = 32`、`device = "cuda"` → `Path(CKPT_PATH).parent.mkdir(...)` → `mallm = Mallm(cfg).to(device)` → `optimizer = torch.optim.AdamW(mallm.parameters(), lr=3e-4)` → 循环 1000 步（zero_grad → get_batch → 前向 → backward → step），每 100 步评估 train/val 并存档 → 循环结束再存一次

**未清理**：`nn`、`F`、`TOKENIZER_PATH`、`TRAIN_TXT`、`VALID_TXT` 这些 import 没用到。

### 4.8 `generate.py`（基本完成）
当前内容（从上到下）：
- `device = "cuda"`
- `checkpoint = torch.load("checkpoints/mallm-12l-8h.pt", map_location=device)`（**路径是硬编码的，指向 8 小时的模型**）
- `cfg = MallmConfig(**checkpoint["config"])` → `mallm = Mallm(cfg).to(device)` → `load_state_dict` → `eval()` → `print(cfg)`
- `tok = Tokenizer.from_file(TOKENIZER_PATH)`、`prompt = "Once upon a time"`、`ids = tok.encode(prompt).ids`、`index = torch.tensor([ids], device=device)`
- `max_ouput_token = 200`（拼写错误，应为 output）、`temperature = 0.6`、`topk = 50`、`eot_id = tok.token_to_id("<|endoftext|>")`
- `with torch.no_grad():` 循环：前向 → `logits[:, -1, :]` → `/ temperature` → `values, _ = torch.topk(logits, topk)` → `logits[logits < values[:, -1:]] = float("-inf")` → softmax → `multinomial` → `out = tok.decode([next_id.item()])` → `torch.cat` 拼接 → 抽到 eot 就 break，否则 `print(out, end="")`

**未做**：上下文截断 `index[:, -cfg.block_size:]`（当前 5+200 < 256 不会崩，但改大 max token 就会报错）；`asdict` import 没用到。

### 4.9 `.gitignore`
包含 `data/`、`checkpoints/`、`runs/`、`.venv`、`__pycache__` 等。`tokenizer-8k.json`（540KB）**保留在 git 里**，因为它定义了所有下游 id 的含义。

---

## 5. 数据

- `data/raw/TinyStoriesV2-GPT4-train.txt`：2,227,753,162 字节（2.1GB），2,717,699 篇故事
- `data/raw/TinyStoriesV2-GPT4-valid.txt`：22,502,601 字节，27,630 篇故事
- 故事之间用**独占一行**的 `<|endoftext|>` 分隔
- valid.txt 的第一篇故事本身就是半截的（以 `u don't have to be scared of the loud dog...` 开头），这是数据集自己的特点，不是 bug

处理产物（已验证）：
- `data/processed/val.bin`：10,961,478 字节 = 5,480,739 tokens × 2 ✅
- `data/processed/train.bin`：1,085,415,226 字节 = 542,707,613 tokens × 2 ✅
- 两者最大 id 都是 8191（= vocab_size - 1，证明 uint16 够用且无溢出）
- 压缩率约 4.11 字节/token

目录结构约定：`data/raw/`（下载来的，只读）、`data/processed/`（脚本产物，可随时删除重建）。

---

## 6. 已有的 checkpoint

`checkpoints/` 目录（gitignore 中）：

1. **`mallm-12l-8h.pt`**（168,687,175 字节）— **最重要的产物**
   - 8 小时训练的结果，约 4215 万参数
   - 配置：`{'vocab_size': 8192, 'block_size': 256, 'd': 512, 'n_head': 8, 'n_layer': 12}`
   - `step: 187627`
   - 键：`model` / `config` / `step`（**本会话中我已把原来的 `model_args` 键名改成了 `config`**，改的时候逐张量比对过、用临时文件 + 原子替换，权重未变）
   - 生成质量很好，能写出情节完整、角色前后一致、带对话的小故事

2. **`mallm.pt`**（55,594,851 字节）
   - 用户自己 `train.py` 跑 1000 步的结果，1389 万参数（d=384, 6 层 6 头）
   - 键：`model` / `config`
   - 生成的是"像英文但读不通"的句子

---

## 7. 8 小时训练的详细结果（重要参考基线）

我当时（用户赶时间要睡觉）直接写的 train.py，配置：
- `BATCH_SIZE=32`、`BLOCK_SIZE=256`、`d=512`、`n_head=8`、`n_layer=12`
- `TRAIN_HOURS=8`（按时间停，不按步数停；学习率的 cosine 衰减也按时间进度算）
- `MAX_LR=5e-4`、`MIN_LR=5e-5`、`WARMUP_STEPS=1000`
- AdamW `betas=(0.9, 0.95)`、`fused=True`
- bf16 autocast、梯度裁剪 1.0
- 每 2000 步存一次，先写 `.tmp` 再 `os.replace`

结果：
- 187,600 步，约 0.154 秒/步，约每秒 5.3 万 token
- 共处理约 15.4 亿 token ≈ 数据集的 2.8 遍
- loss：9.20（step 0）→ 1.90（0.43h）→ 1.47（1.71h）→ 1.35（4.26h）→ 约 1.27（8h）
- 从日志算出的平均：60k-66k 步区间 1.435，最后 6k 步 1.262
- 中间有明显的**平台期**：step 100~400 卡在 5.7~5.9，之后才继续降（先学词频，再学用上下文）

这份训练脚本用户要求删掉了（"这个旧的train不用备份，我来写"），我也删了当时的 `generate.py`。

---

## 8. 本会话的时间线（做了什么）

1. 解释 `open()` 的模式（r/w/a/x、b、+）、默认 encoding、`w` 会立刻清空文件的危险
2. 完成 `prepare.py`：tail flush、`return total`、包成函数、`with open` 提到循环外、`wb` vs `ab` 的幂等性
3. 跑通 prepare，三项验证全过（token 数、文件大小 == total×2、解码回来是通顺英文）
4. 把数据分成 `data/raw/` 和 `data/processed/`，更新 `config.py` 路径（这一步是我代劳的）
5. 讨论空目录 git 不记录的问题 → 结论用 `mkdir(parents=True, exist_ok=True)` 而不是 `.gitkeep`
6. 用户重新写 `train.py` 的 `get_batch`（我先代写过一次，用户后来删掉自己重写）
7. 写 `model.py`：MLP → Attention（单头 → 多头）→ Block → Mallm
8. 讨论各种基础概念（见第 9 节）
9. 加 loss（`cross_entropy`）到 `forward`
10. 写 `train.py` 的训练循环、`evaluate_loss`、`save_ckpt`
11. 引入 `MallmConfig` dataclass，改 `Mallm` 收 config 对象
12. 跑通 1000 步训练：9.20 → 4.23，train 和 val 几乎完全一致（无过拟合）
13. 写 `generate.py`：加载 checkpoint → 编码 prompt → 自回归循环 → 解码
14. 用 8 小时的模型生成，得到高质量故事
15. 把 8h checkpoint 的 `model_args` 键改名为 `config`
16. 加 temperature 和 top-k
17. 开始升级 train.py，**刚讲完 bf16 autocast 的原理，用户还没写**

---

## 9. 已经讲过的知识点（避免重复讲，也便于回顾）

### Python / 工具
- `open()` 的模式：r/w/a/x 主模式、b 后缀、`+` 补上另一半权限；`r+` 原地覆盖 vs `w` 推倒重来；文本模式必须写 `encoding="utf-8"`，二进制模式不能写
- 文本模式会做换行符转换，会破坏二进制数据
- 幂等性：`wb` 每次重来（好），`ab` 追加（重跑会数据翻倍）
- `Path.parent`（单数，一个 Path）vs `.parents`（复数，一个序列）
- `mkdir(parents=True, exist_ok=True)`，`exist_ok` 避免 TOCTOU
- 列表 `append` vs `extend`；`np.stack` vs `np.concatenate` vs `np.array`；Python 列表没有 `.mean()`
- `sum(xs)/len(xs)` 在小列表上比 `np.mean` 更合适（转换开销）
- 参数 vs 全局变量：函数体里只用参数和局部变量；`tok` 这种"每次调用都一样的常量"可以是例外
- 函数内 `lst = []` 是重新绑定名字（外面看不见），`lst.clear()` 才是修改对象
- `if tensor:` 会报错，判断"有没有传"要用 `is not None`
- 类型标注：`typing` 是标准库但现在很少直接用（`list[str]`、`str | None` 已内置）；标注运行时不检查；参数标宽、返回标窄，但不要承诺做不到的事（所以文件句柄标 `io.BufferedWriter` 而不是 `BinaryIO`）
- 命名规范：变量/函数 snake_case、类 CapWords、常量 UPPER_SNAKE；`B`/`T`/`d` 这种大写单字母在张量代码里是公认例外
- dataclass：`@dataclass` 自动生成 `__init__`/`__repr__`；**字段必须有类型标注**，没标注的会被静默忽略；有默认值的字段必须在后面；`asdict()` 转普通字典（存 checkpoint 必须转，因为 `torch.load` 默认只允许基本类型）
- 装饰器就是 `f = decorator(f)` 的语法糖
- `**` 展开字典成具名参数、`*` 展开列表成位置参数；写在函数定义里则是收集
- `.get(key)` 返回 None vs `[key]` 抛 KeyError

### numpy / 文件格式
- uint16 选择理由：vocab 8192 < 65536，2 字节；不写 dtype 会推断成 int64（4 倍体积）；uint8 会溢出（numpy 2.x 的 `np.array` 会报 OverflowError，但 `.astype` 会静默截断）
- `tofile` 只写裸数据，不记录 dtype/shape，是隐式契约；`sep` 参数可以切换成文本模式
- 传文件名给 `tofile` 会覆盖，传句柄才会从光标位置续写
- `tofile` 需要真实文件描述符，`BytesIO` 会报 `UnsupportedOperation: fileno`
- `np.memmap` 不真读文件，用到哪读哪；定长 2 字节 → 第 n 个数在第 2n 字节 → 能随机访问
- `.bin` vs `.pt` vs `.safetensors` 的完整对比（见第 10 节）

### 张量 / PyTorch
- `(B, T, d)` 的含义；B 是行数不是"batch 的数量"；T 是每行的 token 数
- 3 维 `@` = 前面的维度当"份数"，只有最后两维做矩阵乘法
- `transpose(-2, -1)` 用负数下标，多头时才不会错；`.T` 在 3 维上会反转所有维度；`.mT` 是等价简写
- `view` vs `reshape`：view 不复制、不行就报错；reshape 能复制。模型代码里放心用 reshape
- 用整数下标会少一维（`logits[:, -1, :]` → `(1, V)`），用切片会保留（`[:, -1:, :]` → `(1,1,V)`）
- `unsqueeze(0)` / `[None, :]` / 多套一层方括号，都能加 batch 维
- `nn.Module` 的 `.to()` 是原地的，张量的 `.to()` 必须赋值回去
- `nn.ModuleList` vs 普通 list：普通 list 里的子模块不会被登记，`.to()` 搬不走、优化器看不见，且**不报错**
- `model.eval()` / `model.train()` 只是拨 `self.training` 开关；当前模型没有 Dropout/BatchNorm 所以没有实际影响，但要养成习惯；和 `torch.no_grad()` 是两件独立的事
- `state_dict()`（名字→张量，用于存档）vs `parameters()`（只有张量的生成器，用于优化器）
- 存 `state_dict` + config 而不是整个模型对象（后者绑定代码路径，脆弱）
- 权重共享后 `parameters()` 会去重，所以参数量能验证共享是否生效
- `optimizer.zero_grad()` 必须有，因为 PyTorch 默认**累加**梯度（这也是梯度累积能实现的原因）
- `torch.autocast` 的工作方式（矩阵乘法用 bf16、softmax/LayerNorm/cross_entropy 保持 fp32、参数本身仍是 fp32）；只包前向；bf16 不需要 GradScaler 而 fp16 需要

### 模型原理
- Transformer 的数据流；残差流宽度 d 全程不变；只有注意力让 token 互相交流
- Q/K/V 的直觉（我在找什么 / 我是什么 / 我能提供什么）；为什么要三个不同的 Linear
- 点积 → 矩阵乘法一次算完所有两两得分 → `(B,H,T,T)`
- 除以 √d 的作用：防止 softmax 饱和导致梯度消失（原论文的理由）；不除也能跑但会训得差
- 因果掩码：下三角 `torch.tril`、填 `-inf`（填 0 不行，因为 `e^0=1`）；必须在 softmax 之前
- 掩码 ≠ 位置编码：前者管"谁能看谁"，后者管"谁在哪"
- 多头只是 reshape，不增加参数；一个大 Linear 切开 == H 个小 Linear（权重矩阵按行切）；每个头看的是完整的 x，切的是输出
- 拼回时必须先 transpose 再 reshape，否则会把不同 token 的结果接到一起且不报错
- 输出投影（`fc`）的作用：混合各头的结果
- LayerNorm 的公式、和 Linear 的对比（不混合维度、参数只有 2d）、和 BatchNorm 的区别、pre-norm 的位置
- 残差连接：Block 只往传送带上"加"东西，不替换
- Embedding 是查找表，等价于 one-hot × Linear；token id 只是标签，数值大小无意义
- 位置编码：token 向量 + 位置向量（按位相加）；用户问过"为什么不拼接" → 相加是拼接的一般化（模型可以自己学出分区），拼接要人为定比例且会增大 d；进 Block 后第一步是 Linear，`W[c;p] = W₁c + W₂p`，两者表达力接近；RoPE 正是为了让位置和内容不混在一起
- 权重共享（tying）：形状都是 `(vocab, d)`；出口第 i 个输出 = x 和 token i 向量的点积；省 14% 参数；小模型通常更好（罕见 token 能借到出口侧的频繁梯度）；大模型常不共享
- `lm_head` 的命名来源（body/backbone + head）；和 attention head 是同一个词的两个意思
- 训练时一次前向做 T 道题；target 是 input 右移一位
- 交叉熵：softmax → 取正确答案的概率 → `-log(p)`；初始 loss ≈ ln(vocab_size) ≈ 9.01
- `cross_entropy` 默认把第 1 维当类别维，所以必须拍平成 `(B*T, V)` 和 `(B*T,)`；不能自己先 softmax
- 4d 的 MLP 倍数来自原论文；SwiGLU 用 3 个矩阵所以中间层是 8d/3；模型约 2/3 参数在 MLP 里
- 深而瘦 vs 宽而浅：理论依据（万能逼近定理、Telgarsky 2016、Eldan & Shamir 2016、Montúfar 2014 的线性区域计数）、直观解释（帐篷函数套 k 次得 2^k 段）、边界（存在性≠普遍性、能表示≠训得出、Lu 2017 的反向结论、Ba & Caruana 2014）、Transformer 上的实证（Kaplan 2020 说形状影响很弱：6 层 d=4288 和 48 层 d=1600 差 3% 以内；MobileLLM 2024 说小模型更深更好）
- 为什么 MLP 内部不加深：没有残差保护、GPU 更喜欢宽、MLP 宽度像"记忆容量"

### 训练 / 生成
- `get_batch`：随机起点、上限是 `len(data) - T`、x 和 y 错开一位、B 段叠成 `(B, T)`
- B 和 T 的区别：T 决定能看到多少上下文（影响能力上限），B 只影响梯度的稳定性和速度
- block_size 在模型里**只被 pos_emb 用到**，它就是 context window 的硬上限；T ≤ block_size，通常相等
- val loss 要多批取平均（单批波动 ±0.1，而要分辨的差异往往是 0.02-0.05）；评估成本估算（每 100 步 20 批 ≈ 多花 7%）
- AdamW；`loss.backward()` 算梯度存进 `.grad`；`optimizer.step()` 更新
- 学习率写法 `3e-4` = 3×10⁻⁴
- Muon 优化器：把权重矩阵当整体、动量 + Newton-Schulz 近似正交化；只管隐藏层的二维矩阵，embedding/LayerNorm/bias 仍用 AdamW；Kimi K2 用的 MuonClip
- 生成的自回归循环：截断 → 前向 → 取最后位置 → 温度 → top-k → softmax → multinomial → cat
- temperature：除以它再 softmax；<1 更尖锐保守、>1 更平更随机；默认 1；实践中常用 0.6-0.9；推理模型的 API 常常锁死这个参数
- top-k：`torch.topk` 返回 (values, indices)，用 `values[:, -1:]` 当门槛；`[:, -1:]` 保形状才能广播
- top-k 在低温下几乎不生效，但值得一直开：单步 1% 的失误率，200 步下 87% 的故事会崩
- top-p（nucleus）：排序 → 累加 → 到 p 为止；候选数量自适应；实现要点是要把 mask 映射回原位置
- 生成时 lm_head 对所有位置都算是浪费（约 10%），可以只算最后一个位置
- KV cache：跨轮次重复计算是更大的浪费；nanoGPT/vLLM 等的标准做法

---

## 10. 讨论过的"工业界怎么做"（用户很关心这类问题）

- **checkpoint 管理**：一次实验一个文件夹（时间戳/wandb 随机名）+ config.json + 日志；last（用于续训）+ best（用于最终使用）；HF Trainer 用 `checkpoint-{step}` + `save_total_limit`；大模型预训练其实很少用 "best"（数据只看一遍，val loss 单调下降，best≈last），而是高频存档用于故障恢复（Llama 3 405B 54 天被打断 466 次）、保留中间存档用于 loss spike 回滚（PaLM 的做法：退回 100 步前、跳过 200-500 批数据）、最终模型有时是几个存档的权重平均（Llama 3）；Pythia 公开 154 个中间存档、OLMo 也公开数百个
- **配置管理**：nanoGPT 用 dataclass `GPTConfig`；HF 用 `config.json` + `PretrainedConfig`；大框架用 YAML + CLI 覆盖
- **权重格式**：`.pt`（pickle，训练自用）vs `.safetensors`（JSON 头 + 裸数据，发布用，安全、可 mmap、跨语言）；HF 早期的 `pytorch_model.bin` 其实是 pickle（后缀不代表格式）；safetensors 不允许共享张量，所以 tied weight 要只存一个
- **实验记录**：wandb 是事实标准，TensorBoard/SwanLab 可本地；LLM 特别要记 grad norm（loss 爆炸前常先有尖峰）、tokens/s、定期生成样本
- **现代架构**（DeepSeek V3 / Kimi K2）：骨架还是 decoder-only Transformer；换掉的零件是 RMSNorm、RoPE、MLA（压缩 KV cache）、MoE（DeepSeek 256 专家选 8 + 1 共享；Kimi 384 专家选 8 + 1 共享）；Kimi K2 基本照搬 DeepSeek V3 的结构 + Muon 优化器；后续方向是稀疏注意力、线性注意力混合
- **推理优化**：prefill/decode 分离、连续批处理、PagedAttention、投机解码、量化
- **GPT-2 词表 50257 → 50304 提速 25%** 的例子（对齐到 64 的倍数）
- **tokenizer 对性能的影响**：词表大小的权衡、和数据不匹配时伤害最大、数字/空格/字母级任务受切分方式影响、"故障 token"、不同 tokenizer 的 loss 不可直接比较（要用 bits per byte）

---

## 11. 算过的数字（可直接引用）

- 调试配置（d=384, 6 层 6 头, vocab 8192, block 256）：**13,891,584 参数**（已验证）
  - tok_emb 3,145,728 + pos_emb 98,304 + 6×Block 10,646,784 + ln_final 768
- 8h 配置（d=512, 12 层 8 头）：**42,155,008 参数**（已验证，checkpoint 168MB = 42.15M × 4 字节）
- 8 层 d=512：约 29,676,544（共享权重后）/ 33,870,848（不共享）
- 每层参数 ≈ 12d²（注意力 4d² + MLP 8d²）
- 训练显存 ≈ 参数量 × 16 字节（fp32 权重 4 + 梯度 4 + Adam 的 m/v 各 4）
- 各规模在这台机器上跑完一遍 542.7M token 的估算：1400 万→1 小时、2700 万→2 小时、4200 万→3 小时、1.1 亿→7.5 小时
- Chinchilla：每参数约 20 token；5.43 亿 token ↔ 约 2700 万参数
- 重复训练的研究结论：重复到约 4 遍，效果接近同量的新数据

---

## 12. 踩过的坑（本会话 + 之前）

- `masked_fill` 返回新张量，不赋值回去等于没做（静默）
- 函数内 `storys = []` 清不掉调用方的 list（静默，导致 O(n²) 和数据重复）
- 清空 buffer 的位置：必须和 flush 成对，放 if 外面会导致一个字节都不写
- `nn.modules` / `nn.module` / `nn.layerNorm` 大小写错误
- `nn.GELU(x)` 是构造零件不是调用；零件要在 `__init__` 造、`forward` 用
- `q @ k` 忘记转置：T=512、d=512 时**形状恰好合法**，不报错但语义全错
- `logits[:, 1, :]` 写成 1 而不是 -1
- 生成循环忘记 `torch.cat`（没有自回归）
- `mallm.parameters` 少括号传给优化器
- `valid_losses.mean()`（列表没有这个方法）
- `evaluate_loss` 里 `return` 缩进在循环内 → 只算 1 批
- 训练循环里 `loss = evaluate_loss(...)` 覆盖了张量 → `.item()` 报错
- `"model": mallm.parameters`（应为 `state_dict()`）
- `Mallm(asdict(MallmConfig))`：方向反了 + 传的是类不是实例
- `MallmConfig()` 空类体 → IndentationError
- `block_size` 默认值误写成 32
- `torch.save` 的字典里存 dataclass 对象会导致 `torch.load(weights_only=True)` 拒绝
- 数据目录分层后忘记 `mkdir` → `open(..., "wb")` 不会自动建目录
- `open(VALID_TXT, encoding="utf-8", mode="wb")`：差点清空 22MB 语料，被 `ValueError: binary mode doesn't take an encoding argument` 救了（这个检查发生在打开文件之前）
- 旧 checkpoint 用 `model_args` 键、新的用 `config` 键（已通过改文件统一）

---

## 13. 下一步计划

### 正在做：升级 `train.py`（按顺序，每项加完跑几百步验证）

1. **bf16 混合精度**（刚讲完原理，用户还没写）
   - 把前向包进 `with torch.autocast(device, dtype=torch.bfloat16):`
   - 只包前向，`backward()` 和 `step()` 留在外面
   - 验证方式：`time uv run python train.py` 对比前后耗时，loss 曲线形状应该不变
2. **梯度裁剪**：`torch.nn.utils.clip_grad_norm_(mallm.parameters(), 1.0)`，放在 backward 之后、step 之前
3. **学习率预热 + cosine 衰减**：写一个 `get_lr(step)`，每步 `for g in optimizer.param_groups: g["lr"] = lr`
4. **last / best 两个存档**：best 只在 val loss 创新低时覆盖
5. 可选：把 `1000`（步数）、`100`（评估间隔）、`20`（评估批数）、`3e-4`（lr）提成变量

### 之后

- 用升级后的脚本正式训练一次（3-8 小时），得到用户自己的好模型
- `generate.py` 补上下文截断、加 top-p、考虑做 temperature 对比实验（`torch.manual_seed(42)` 固定种子）
- 推理优化：`lm_head` 只算最后位置（3 行）、KV cache（改动较大，能加深对注意力的理解）
- 模型现代化（逐个替换、每次对比 loss）：RMSNorm → RoPE → SwiGLU；把手写注意力换成 `F.scaled_dot_product_attention`（省显存，能支持更大 batch/context）
- 后训练：SFT（用 TinyStories-Instruct，用到预留的 `<|user|>` / `<|assistant|>`）→ DPO → 可选 GRPO（用"指定词有没有用上""长度合不合规"这类可自动验证的奖励）
- LoRA 可以在 SFT 或 DPO 阶段顺便试（本模型全量微调毫无压力，LoRA 主要是学习用）

### 明确讨论过但决定**暂不做**的

- 双语（中文）支持：需要重训 tokenizer（词表提到 16384）、重跑 prepare、从头训练；建议先把英文版整条链路走完
- 用户问过"最终成品能问问题吗" → **不能**，TinyStories 里没有世界知识；成品是"命题作文机器"；想要能对话的模型应该走"微调现成小模型（Qwen3 0.6B / Llama 3.2 1B）+ LoRA"这条路
- `check_tokenizer.py` 剩下的 4 项检查

---

## 14. 对话风格备忘

- 用户用中文提问，回答也用中文
- 用户经常问"这个是什么意思"、"再详细易懂一点"、"举例子"、"慢点" → 要用具体的小数字走一遍（比如 6 个词的迷你词表、`data = np.arange(100, 120)` 这种假数据）
- 用户关心"工业界/最新模型是怎么做的"，这类问题要给具体的模型名和做法，不确定的地方要明说（我的知识截止 2026 年 5 月）
- 用户会质疑我的说法（比如"这个不能算碰巧，这是我们这么设计的"、"为什么 array 更短？"），我说错了要直接承认并更正，不要过度道歉
- 每次 review 后要明确给出"下一步做什么"
