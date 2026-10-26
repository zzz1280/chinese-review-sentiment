# -*- coding: utf-8 -*-
"""03_finetune_bert.py — 微调中文 RoBERTa (hfl/chinese-roberta-wwm-ext) 情感分类
自动检测可用显存：GPU 空闲(>2.5GB)则全量训练，否则退回 CPU 子集训练。
结束后在测试集与外卖评论(跨领域)上评估。"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

MODE = 'CPU'
if torch.cuda.is_available():
    free_b, _ = torch.cuda.mem_get_info()
    print(f'显存空闲: {free_b / 1024**2:.0f} MiB')
    if free_b / 1024**2 > 2500:
        MODE = 'GPU'
print(f'运行模式: {MODE}')

if MODE == 'GPU':
    N_TRAIN, N_TEST, N_WAIMAI, EPOCHS, BS, EVAL_BS, FP16 = 999999, 999999, 999999, 2, 32, 64, True
else:
    N_TRAIN, N_TEST, N_WAIMAI, EPOCHS, BS, EVAL_BS, FP16 = 8000, 2000, 1000, 2, 16, 32, False

df = pd.read_csv('data/online_shopping_10_cats.csv').dropna(subset=['review'])
idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df['cat'])
if MODE == 'CPU':
    idx_train, _ = train_test_split(idx_train, train_size=N_TRAIN, random_state=42,
                                    stratify=df.loc[idx_train, 'label'])
    idx_test, _ = train_test_split(idx_test, train_size=N_TEST, random_state=42,
                                   stratify=df.loc[idx_test, ['cat', 'label']])
print(f'训练 {len(idx_train)} 条 / 测试 {len(idx_test)} 条')

tok = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

def make_ds(texts, labels):
    d = pd.DataFrame({'review': list(texts), 'labels': list(labels)}).reset_index(drop=True)
    ds = Dataset.from_pandas(d, preserve_index=False)
    return ds.map(lambda b: tok(b['review'], truncation=True, max_length=128),
                  batched=True, remove_columns=['review'])

train_ds = make_ds(df.loc[idx_train, 'review'], df.loc[idx_train, 'label'])
test_ds = make_ds(df.loc[idx_test, 'review'], df.loc[idx_test, 'label'])

model = AutoModelForSequenceClassification.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=2)

args = TrainingArguments(
    output_dir='models/roberta-sentiment',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=EVAL_BS,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='no',
    logging_steps=20,
    fp16=FP16,
    report_to='none',
    seed=42,
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {'accuracy': float((preds == p.label_ids).mean()),
            'f1': float(f1_score(p.label_ids, preds))}

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
trainer.train()

results = {'mode': MODE, '训练条数': len(idx_train), '测试条数': len(idx_test)}
ev = trainer.evaluate(test_ds)
results['测试集'] = {k: round(v, 4) for k, v in ev.items() if k.startswith('eval_')}

# 跨领域：外卖评论
waimai = pd.read_csv('data/waimai_10k.csv').dropna(subset=['review'])
if len(waimai) > N_WAIMAI:
    waimai = waimai.sample(N_WAIMAI, random_state=42).reset_index(drop=True)
w_ds = make_ds(waimai['review'], waimai['label'])
ev_w = trainer.evaluate(w_ds)
results['跨领域_外卖'] = {k: round(v, 4) for k, v in ev_w.items() if k.startswith('eval_')}

results['训练日志'] = trainer.state.log_history

os.makedirs('results', exist_ok=True)
with open('results/finetune_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

model.save_pretrained('models/roberta-sentiment')
tok.save_pretrained('models/roberta-sentiment')
print('最终结果:', json.dumps({k: v for k, v in results.items() if k != '训练日志'}, ensure_ascii=False))
print('完成')
