# 中文购物评论情感分析：从 EDA 到少样本跨领域迁移

用 6.2 万条中文购物评论走完 NLP 实战全流程：**数据审计 → 探索分析 → 基线模型 → BERT 微调 → 跨领域迁移 → 多种子检验**。每条结论都附不确定性分析，所有实验脚本可一键复现。

## 核心结果

| 实验 | 准确率 |
|---|---|
| TF-IDF + 逻辑回归（基线） | 90.07% |
| RoBERTa 微调（hfl/chinese-roberta-wwm-ext） | **95.42%** |
| 跨领域零适应（购物模型 → 外卖评论） | 87.74% |
| +100 条外卖标注 · 顺序迁移 | 89.53% ± 0.17% |
| +100 条外卖标注 · 无迁移对照 | 81.34% ± 3.51% |

（±为 6 次独立微调的标准差：2 次数据抽样 × 3 个训练种子）

## 训练好的模型

已上传 Hugging Face：[zhuhua666/chinese-review-sentiment-roberta](https://huggingface.co/zhuhua666/chinese-review-sentiment-roberta)（Model Card 含结果、训练细节与局限性说明）

```python
from transformers import pipeline

clf = pipeline('text-classification', model='zhuhua666/chinese-review-sentiment-roberta')
clf('质量非常好，物流也快，下次还来')   # LABEL_1=好评 / LABEL_0=差评
```

## 亮点发现

1. **数据集有标签噪声**：「不错，挺实惠的」等明显好评被标为差评——模型的上限由数据质量决定，90%+ 的词袋基线已接近该数据的标签质量天花板。
2. **领域偏移对词袋和 BERT 同样致命**：跨领域（购物→外卖）两者都掉约 8 个点，换模型架构解决不了分布变化。
3. **迁移先验 = 提分器 + 稳定器**：100 条标注场景下，先在购物评论上训练过的模型不仅均值高 8 个点，方差还是对照组的 1/20；对照组最差一次跑出 75.1%（比零适应还差）——小数据微调不注入先验就是在赌运气。
4. **100 条标注拿回大部分损失**：87.7% → 89.5%，之后边际收益快速递减；N=2000 时迁移优势完全消失。标注预算的实操答案：先标 100 条看曲线斜率。
5. **方差来源分解**：小样本场景下，「抽到哪 100 条」的影响（2.6 个点）大于随机种子的影响——数据抽样比 seed 更值得控制。

## 快速开始

```bash
# 1. 下载数据（国内网络走 hf-mirror）
curl -L --create-dirs -o data/online_shopping_10_cats.csv \
  "https://hf-mirror.com/datasets/XiangPan/online_shopping_10_cats_62k/resolve/main/online_shopping_10_cats.csv"
curl -L --create-dirs -o data/waimai_10k.csv \
  "https://hf-mirror.com/datasets/XiangPan/waimai_10k/resolve/main/waimai_10k.csv"

# 2. 安装依赖（本机环境：Python 3.12 / torch 2.7.1+cu118 / transformers 5.20）
pip install jieba wordcloud scikit-learn pandas matplotlib joblib \
            "transformers" "datasets" gradio torch

# 3. 复现实验（顺序执行）
python scripts/01_eda.py              # 探索分析，出 6 张图（约 2 分钟）
python scripts/02_baseline.py         # 基线模型 + 跨领域测试（约 2 分钟）
python scripts/03_finetune_bert.py    # BERT 微调，自动检测 GPU（约 10 分钟）
python scripts/04_plot_training.py    # 训练曲线与模型对比图
python scripts/05_fewshot_transfer.py # 少样本迁移曲线（约 5 分钟，断点续跑）
python scripts/06_multiseed.py        # 多种子×多次抽样检验（24 次微调，可分批）

# 4. 试用模型（Gradio 网页）
python app.py   # http://127.0.0.1:7860
```

无 GPU 时 `03_finetune_bert.py` 自动退回 CPU 子集训练；国内网络请设置 `HF_ENDPOINT=https://hf-mirror.com`（脚本已内置）。

## 项目结构

```
├── scripts/
│   ├── 01_eda.py               # 品类差评率 / 长度分布 / 高低频词 / 词云
│   ├── 02_baseline.py          # TF-IDF+逻辑回归，分品类准确率 + 跨领域测试
│   ├── 03_finetune_bert.py     # RoBERTa 微调（GPU/CPU 自适应）
│   ├── 04_plot_training.py     # 训练曲线、模型对比图
│   ├── 05_fewshot_transfer.py  # 少样本跨领域迁移学习曲线（断点续跑）
│   └── 06_multiseed.py         # 2 抽样 × 3 种子不确定性量化（断点续跑）
├── figures/                    # 全部图表（EDA 6 张 + 训练/迁移/误差棒图）
├── results/                    # 各实验指标 JSON
├── report.md                   # 完整分析报告（推荐从这里开始读）
└── app.py                      # Gradio 演示
```

## 数据来源与许可

- 数据来自 [ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus) 整理的公开评论数据（`online_shopping_10_cats` 62,774 条、`waimai_10k` 11,987 条），版权归原作者，本项目仅用于研究学习，原始数据不入库。
- 代码部分以 [MIT License](LICENSE) 发布。
- 模型底座：[hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)。
