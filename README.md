# mini-bpe

字节级 BPE（Byte Pair Encoding）分词器。GPT-2 / GPT-4 / Llama 的 tokenizer 都是这个算法（OpenAI 的 tiktoken、HuggingFace 的 tokenizers，核心都是用 Rust 写的）。

## 依赖

`clap`（derive）· `serde`（derive）· `serde_json`。**不能用** `tiktoken-rs`、`tokenizers` 等现成的分词 crate。

## 算法

**训练**：

1. 把语料按 UTF-8 拆成字节，每个字节就是一个 token，id 是 0–255
2. 数所有**相邻**两个 token 一起出现了多少次
3. 出现最多的那一对，合并成一个新 token，id = 256、257、258……依次往上
4. 把语料里所有这一对（从左往右，不重叠）替换成新 id，回到第 2 步
5. 词表够大了（256 + 合并次数 = `--vocab-size`），或者出现最多的一对只出现 1 次，就停

**并列**：出现次数一样的，选 `(a, b)` 最小的那对（先比 a，再比 b）。这条规则保证所有人训练出来的结果一模一样，测试才能对答案。

**编码**一段新文本：先拆成字节；然后反复找「在合并规则里排得最靠前」的相邻对，合并它，直到没有可以合并的。

**解码**：每个 id 查表换回它的字节串，拼起来，再转回 UTF-8 文本。

例子（维基百科同款）：`aaabdaaabac` 训练到 259

```
第 1 次：aa 出现 4 次最多      → 256 = (97, 97)    256 a b d 256 a b a c
第 2 次：(256,97)、(97,98) 都是 2 次，并列取小的 → 257 = (97, 98)
第 3 次：(256,257) 2 次        → 258 = (256, 257)
编码 aaabdaaabac → 258 100 258 97 99
```

## 用法

```
mini-bpe train <CORPUS> [--vocab-size 512] [-o model.json]
mini-bpe encode -m <MODEL> [TEXT]            # 不给 TEXT 读标准输入
mini-bpe decode -m <MODEL> <IDS>...
mini-bpe vocab  -m <MODEL> [--top 10]        # 看最后学到的几个 token
```

## 要求

1. **模型文件格式**：`{"merges": [[97, 97], [97, 98], [256, 257]]}`，第 i 条合并产生的 id 是 256 + i
2. **train 输出一行**：`合并 N 次，词表 V；语料 B 字节 → T 个 token（压缩到 P%）`（这行不测格式）
3. **encode 输出**：id 用空格隔开，一行；空文本输出空行
4. **decode 输出**：还原的文本
5. **往返**：任何文本（中文、emoji、训练时没见过的句子）`decode(encode(x)) == x`
6. **出错退出码 1**：`--vocab-size` 小于 256 / decode 给了不存在的 id / 模型文件不存在、格式不对、合并规则引用了还不存在的 id
7. `vocab` 输出格式自定（不测）

## 提示

- 数相邻对：`ids.windows(2)` + `HashMap<(u32, u32), usize>` + `entry().or_insert(0)`（第七讲讲过）
- 不是每个 token 都是完整的汉字（一个汉字 3 个字节，可能被拆开）。`vocab` 打印时用 `String::from_utf8_lossy`
- 用前几讲的讲义（几十 KB）训练到 1000，参考实现 1 秒内跑完。你的如果要几分钟，想想哪里重复算了

## 选做（不加分，期末能接着用）

- 下载 GPT-2 的 `vocab.bpe`，读进来，和 Python 的 `tiktoken` 逐个 token 对比结果
- 训练时先按空格和标点把语料切成词，词与词之间不合并（GPT-2 就是这么做的）
