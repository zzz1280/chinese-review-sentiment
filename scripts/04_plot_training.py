# -*- coding: utf-8 -*-
"""04_plot_training.py — 绘制 BERT 微调训练曲线与「基线 vs BERT」对比图
需先成功运行 03_finetune_bert.py。"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

if not os.path.exists('results/finetune_metrics.json'):
    print('尚未运行 03_finetune_bert.py（results/finetune_metrics.json 不存在），跳过绘图')
    raise SystemExit

r = json.load(open('results/finetune_metrics.json', encoding='utf-8'))
logs = [l for l in r['训练日志'] if 'loss' in l]
evals = [l for l in r['训练日志'] if 'eval_loss' in l]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot([l['step'] for l in logs], [l['loss'] for l in logs], lw=1)
axes[0].set_title('训练损失曲线'); axes[0].set_xlabel('step'); axes[0].set_ylabel('loss')
if evals:
    ep = [e['epoch'] for e in evals]
    axes[1].plot(ep, [e['eval_accuracy'] for e in evals], marker='o', label='accuracy')
    axes[1].plot(ep, [e['eval_f1'] for e in evals], marker='s', label='f1')
    axes[1].set_title('测试集指标'); axes[1].set_xlabel('epoch'); axes[1].legend()
plt.tight_layout(); plt.savefig('figures/bert_training.png', dpi=150); plt.close()

base = json.load(open('results/baseline_metrics.json', encoding='utf-8'))
names = ['测试集准确率', '跨领域准确率\n(购物→外卖)']
base_vals = [base['accuracy'], base['跨领域_waimai_accuracy']]
bert_vals = [r['测试集']['eval_accuracy'], r['跨领域_外卖']['eval_accuracy']]

fig, ax = plt.subplots(figsize=(7.5, 4.8))
x = range(len(names))
ax.bar([i - 0.18 for i in x], base_vals, 0.36, label='TF-IDF + 逻辑回归', color='#4a89dc')
ax.bar([i + 0.18 for i in x], bert_vals, 0.36, label='RoBERTa 微调', color='#d64541')
for i, v in enumerate(base_vals):
    ax.text(i - 0.18, v + 0.004, f'{v:.3f}', ha='center', fontsize=10)
for i, v in enumerate(bert_vals):
    ax.text(i + 0.18, v + 0.004, f'{v:.3f}', ha='center', fontsize=10)
ax.set_xticks(list(x)); ax.set_xticklabels(names)
ax.set_ylim(0.7, 1.0); ax.legend()
ax.set_title('基线模型 vs 微调 RoBERTa')
plt.tight_layout(); plt.savefig('figures/model_compare.png', dpi=150); plt.close()
print('已生成 figures/bert_training.png 和 figures/model_compare.png')
