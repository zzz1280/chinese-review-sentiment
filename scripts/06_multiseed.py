# -*- coding: utf-8 -*-
"""06_multiseed.py — 关键点位多种子×多次抽样实验
量化少样本迁移结论的不确定性：
  N ∈ {100, 2000} × A/B 两组 × 2 次数据抽样 × 3 个训练种子 = 24 次微调
A/B 两组共用同一批抽样（配对比较）。逐次保存、断点续跑，结束时输出聚合与图。
用法（后台任务有 10 分钟上限，按点位分批跑）：
  python scripts/06_multiseed.py --ns 100            # N=100 双组
  python scripts/06_multiseed.py --ns 2000 --arms A  # N=2000 A 组
  python scripts/06_multiseed.py --ns 2000 --arms B  # N=2000 B 组（完成后出聚合图）
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument('--ns', type=int, nargs='+', default=[100, 2000])
ap.add_argument('--arms', nargs='+', default=['A', 'B'], choices=['A', 'B'])
ap.add_argument('--resamples', type=int, nargs='+', default=[0, 1])
args = ap.parse_args()
print(f'本轮: N={args.ns} 组={args.arms} 抽样={args.resamples}')

SHOP_DIR = 'models/roberta-sentiment'
RAW_NAME = 'hfl/chinese-roberta-wwm-ext'
RESAMPLES = {0: 42, 1: 1234}   # 抽样编号 → random_state（0 号与 05 脚本同分布）
SEEDS = [42, 43, 44]
ARMS = {'A': ('顺序迁移(购物→外卖)', SHOP_DIR), 'B': ('无迁移对照(原始RoBERTa)', RAW_NAME)}

df = pd.read_csv('data/waimai_10k.csv').dropna(subset=['review'])
idx_pool, idx_test = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df['label'])
print(f'微调池 {len(idx_pool)} / 留出测试 {len(idx_test)}')

tok = AutoTokenizer.from_pretrained(SHOP_DIR)
collator = DataCollatorWithPadding(tok)

def make_ds(texts, labels):
    d = pd.DataFrame({'review': list(texts), 'labels': list(labels)}).reset_index(drop=True)
    ds = Dataset.from_pandas(d, preserve_index=False)
    return ds.map(lambda b: tok(b['review'], truncation=True, max_length=128),
                  batched=True, remove_columns=['review'])

test_ds = make_ds(df.loc[idx_test, 'review'], df.loc[idx_test, 'label'])

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {'accuracy': float((preds == p.label_ids).mean()),
            'f1': float(f1_score(p.label_ids, preds))}

def run_one(load_path, arm_key, n, rs_id, seed, train_ds):
    model = AutoModelForSequenceClassification.from_pretrained(load_path, num_labels=2)
    targs = TrainingArguments(
        output_dir='models/tmp',
        num_train_epochs=5 if n <= 500 else 3,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        learning_rate=2e-5, weight_decay=0.01,
        save_strategy='no', logging_steps=999,
        fp16=torch.cuda.is_available(), report_to='none', seed=seed)
    t = Trainer(model=model, args=targs, train_dataset=train_ds,
                data_collator=collator, compute_metrics=compute_metrics)
    t.train()
    out = t.evaluate(test_ds)
    del t, model
    torch.cuda.empty_cache()
    return {'n': n, 'arm': arm_key, 'resample': rs_id, 'seed': seed,
            'accuracy': round(out['eval_accuracy'], 4), 'f1': round(out['eval_f1'], 4)}

os.makedirs('results', exist_ok=True)
results = []
if os.path.exists('results/multiseed_metrics.json'):
    results = json.load(open('results/multiseed_metrics.json', encoding='utf-8'))['结果']
done = {(r['n'], r['arm'], r['resample'], r['seed']) for r in results}

for n in args.ns:
    for rs_id in args.resamples:
        rs = RESAMPLES[rs_id]
        sub, _ = train_test_split(idx_pool, train_size=n, random_state=rs,
                                  stratify=df.loc[idx_pool, 'label'])
        train_ds = make_ds(df.loc[sub, 'review'], df.loc[sub, 'label'])
        for arm_key in args.arms:
            _, path = ARMS[arm_key]
            for seed in SEEDS:
                if (n, arm_key, rs_id, seed) in done:
                    continue
                row = run_one(path, arm_key, n, rs_id, seed, train_ds)
                results.append(row)
                print('完成:', row, flush=True)
                with open('results/multiseed_metrics.json', 'w', encoding='utf-8') as f:
                    json.dump({'结果': results}, f, ensure_ascii=False, indent=2)

# ---- 聚合与图 ----
res_df = pd.DataFrame(results)
agg = res_df.groupby(['n', 'arm'])['accuracy'].agg(['mean', 'std', 'count'])
print('\n=== 按点位聚合（accuracy） ===')
print(agg.round(4))
paired = res_df.groupby(['n', 'resample', 'arm'])['accuracy'].mean().unstack('arm')
if {'A', 'B'} <= set(paired.columns):
    paired['A-B配对差'] = paired['A'] - paired['B']
    print('\n=== 同抽样配对差（种子均值） ===')
    print(paired.round(4))

fig, ax = plt.subplots(figsize=(7, 5))
ns_plot = sorted(res_df['n'].unique())
width = 0.35
for i, arm_key in enumerate(['A', 'B']):
    means, stds = [], []
    for n in ns_plot:
        accs = res_df[(res_df.n == n) & (res_df.arm == arm_key)]['accuracy']
        means.append(accs.mean())
        stds.append(accs.std() if len(accs) > 1 else 0)
    xs = [j + (i - 0.5) * width for j in range(len(ns_plot))]
    ax.bar(xs, means, width, yerr=stds, capsize=5,
           color='#d64541' if arm_key == 'A' else '#4a89dc',
           label='顺序迁移（购物模型）' if arm_key == 'A' else '无迁移对照（原始RoBERTa）')
    for x, m in zip(xs, means):
        ax.text(x, m + 0.004, f'{m:.3f}', ha='center', fontsize=9)
try:
    z = [r['accuracy'] for r in json.load(open('results/fewshot_metrics.json', encoding='utf-8'))['结果']
         if r['tag'] == '购物模型零适应'][0]
    ax.axhline(z, ls='--', c='gray', lw=1, label=f'零适应参照 {z:.3f}')
except Exception:
    pass
ax.set_xticks(range(len(ns_plot)))
ax.set_xticklabels([f'N={n}' for n in ns_plot])
ax.set_ylabel('外卖留出测试集准确率')
ax.set_ylim(0.75, 0.95)
ax.set_title('多种子×多次抽样（每柱最多 6 次运行，均值±标准差）')
ax.legend()
plt.tight_layout(); plt.savefig('figures/fewshot_multiseed.png', dpi=150); plt.close()
print('\n图已保存 figures/fewshot_multiseed.png')
