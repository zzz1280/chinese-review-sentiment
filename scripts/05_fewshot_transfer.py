# -*- coding: utf-8 -*-
"""05_fewshot_transfer.py — 少样本跨领域迁移实验
问题：在购物评论上训练好的模型，只给 N 条外卖评论的标注数据，能适应新领域到什么程度？

实验组：
  参照  购物模型零适应（N=0，直接在留出测试集上评估）
  A组   顺序迁移：购物模型 → N 条外卖样本微调，N ∈ {100, 500, 1000, 2000}
  B组   对照：原始预训练模型（不经购物训练）→ 1000 条外卖样本微调
评估：统一在 waimai 20% 分层留出测试集（random_state=42）。
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

SHOP_DIR = 'models/roberta-sentiment'
RAW_NAME = 'hfl/chinese-roberta-wwm-ext'
NS_A = [100, 500, 1000, 2000]
NS_B = [100, 1000, 2000]

df = pd.read_csv('data/waimai_10k.csv').dropna(subset=['review'])
idx_pool, idx_test = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df['label'])
print(f'外卖微调池 {len(idx_pool)} 条 / 留出测试 {len(idx_test)} 条')

tok = AutoTokenizer.from_pretrained(SHOP_DIR)

def make_ds(texts, labels):
    d = pd.DataFrame({'review': list(texts), 'labels': list(labels)}).reset_index(drop=True)
    ds = Dataset.from_pandas(d, preserve_index=False)
    return ds.map(lambda b: tok(b['review'], truncation=True, max_length=128),
                  batched=True, remove_columns=['review'])

test_ds = make_ds(df.loc[idx_test, 'review'], df.loc[idx_test, 'label'])
collator = DataCollatorWithPadding(tok)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {'accuracy': float((preds == p.label_ids).mean()),
            'f1': float(f1_score(p.label_ids, preds))}

def eval_model(model):
    t = Trainer(model=model, compute_metrics=compute_metrics, data_collator=collator,
                args=TrainingArguments(output_dir='models/tmp', per_device_eval_batch_size=64,
                                       report_to='none', seed=42))
    out = t.evaluate(test_ds)
    del t
    return round(out['eval_accuracy'], 4), round(out['eval_f1'], 4)

def finetune_and_eval(load_path, n, tag):
    sub, _ = train_test_split(idx_pool, train_size=n, random_state=42,
                              stratify=df.loc[idx_pool, 'label'])
    train_ds = make_ds(df.loc[sub, 'review'], df.loc[sub, 'label'])
    model = AutoModelForSequenceClassification.from_pretrained(load_path, num_labels=2)
    args = TrainingArguments(
        output_dir='models/tmp',
        num_train_epochs=5 if n <= 500 else 3,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        learning_rate=2e-5, weight_decay=0.01,
        save_strategy='no', logging_steps=10,
        fp16=torch.cuda.is_available(), report_to='none', seed=42)
    t = Trainer(model=model, args=args, train_dataset=train_ds,
                data_collator=collator, compute_metrics=compute_metrics)
    t.train()
    out = t.evaluate(test_ds)
    acc, f1 = round(out['eval_accuracy'], 4), round(out['eval_f1'], 4)
    del t, model
    torch.cuda.empty_cache()
    return {'tag': tag, 'n': n, 'accuracy': acc, 'f1': f1}

rows = []
if os.path.exists('results/fewshot_metrics.json'):
    rows = json.load(open('results/fewshot_metrics.json', encoding='utf-8'))['结果']
done = {(r['tag'], r['n']) for r in rows}

# 参照：购物模型零适应
if ('购物模型零适应', 0) not in done:
    model = AutoModelForSequenceClassification.from_pretrained(SHOP_DIR)
    acc, f1 = eval_model(model)
    del model
    torch.cuda.empty_cache()
    rows.append({'tag': '购物模型零适应', 'n': 0, 'accuracy': acc, 'f1': f1})
    print('零适应:', rows[-1])

# A 组：顺序迁移
for n in NS_A:
    if ('顺序迁移(购物→外卖)', n) not in done:
        rows.append(finetune_and_eval(SHOP_DIR, n, '顺序迁移(购物→外卖)'))
        print('A组', rows[-1])

# B 组：原始预训练模型直接微调（对照组）
for n in NS_B:
    if ('预训练模型直接微调(无迁移)', n) not in done:
        rows.append(finetune_and_eval(RAW_NAME, n, '预训练模型直接微调(无迁移)'))
        print('B组', rows[-1])

os.makedirs('results', exist_ok=True)
with open('results/fewshot_metrics.json', 'w', encoding='utf-8') as f:
    json.dump({'测试集条数': len(idx_test), '结果': rows}, f, ensure_ascii=False, indent=2)

# ---- 学习曲线 ----
zero = [r for r in rows if r['tag'] == '购物模型零适应'][0]
arm_a = sorted([r for r in rows if r['tag'].startswith('顺序迁移')], key=lambda r: r['n'])
arm_b = sorted([r for r in rows if r['tag'].startswith('预训练模型直接')], key=lambda r: r['n'])

fig, ax = plt.subplots(figsize=(8, 5))
xs = [0] + [r['n'] for r in arm_a]
ys = [zero['accuracy']] + [r['accuracy'] for r in arm_a]
ax.plot(xs, ys, marker='o', lw=2, color='#d64541', label='顺序迁移（购物模型→外卖微调）')
if len(arm_b) > 1:
    ax.plot([r['n'] for r in arm_b], [r['accuracy'] for r in arm_b], marker='s', lw=2,
            color='#4a89dc', label='无迁移对照（原始RoBERTa直接微调）')
else:
    ax.scatter([arm_b[0]['n']], [arm_b[0]['accuracy']], marker='s', s=90, color='#4a89dc', zorder=5)
ax.axhline(zero['accuracy'], ls='--', c='gray', lw=1, label=f"零适应上限 {zero['accuracy']:.3f}")
for x, y in zip(xs[1:], ys[1:]):
    ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)
for r in arm_b:
    ax.annotate(f"{r['accuracy']:.3f}", (r['n'], r['accuracy']), textcoords='offset points',
                xytext=(0, -14), ha='center', fontsize=9)
ax.set_xscale('symlog')
ax.set_xticks(xs)
ax.set_xticklabels([0, 100, 500, 1000, 2000])
ax.set_xlabel('外卖领域标注样本数 N'); ax.set_ylabel('外卖留出测试集准确率')
ax.set_title('少样本跨领域迁移学习曲线'); ax.legend(loc='lower right', fontsize=9)
plt.tight_layout(); plt.savefig('figures/fewshot_curve.png', dpi=150); plt.close()
print('完成，图已存 figures/fewshot_curve.png')
