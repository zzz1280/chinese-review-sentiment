# -*- coding: utf-8 -*-
"""01_eda.py — 购物评论情感数据集探索性分析"""
import json
import os

import jieba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA = 'data/online_shopping_10_cats.csv'
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

STOP = set("""
的 了 是 我 你 他 她 它 我们 你们 他们 也 都 就 不 很 人 一 在 有 和 与 及 为 这 那
之 还 被 把 让 给 但 而 等 又 或 啊 吧 呢 吗 呀 哦 嗯 哈 唉 非常 就是 觉得 感觉 还是
已经 可以 没有 什么 这个 那个 自己 东西 时候 因为 所以 但是 而且 如果 虽然 然后 只是
一直 特别 更加 有点 一些 一次 然后 还是 一个 不是 就是 就是说 真的 感到 而且 另外
""".split())

def tokens(text):
    return [w for w in jieba.lcut(str(text)) if len(w) > 1 and w not in STOP and not w.isdigit()]

df = pd.read_csv(DATA).dropna(subset=['review'])
df['len'] = df['review'].str.len()

summary = {'总条数': int(len(df)), '正评': int((df.label == 1).sum()), '差评': int((df.label == 0).sum())}
print(summary)

# ---- 1. 各品类差评率 ----
cat_stats = df.groupby('cat').agg(条数=('label', 'size'), 差评率=('label', lambda s: 1 - s.mean()))
cat_stats = cat_stats.sort_values('差评率', ascending=False)
print(cat_stats)
summary['各品类差评率'] = cat_stats['差评率'].round(4).to_dict()

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(cat_stats.index, cat_stats['差评率'] * 100,
              color=['#d64541' if r > 0.5 else '#4a89dc' for r in cat_stats['差评率']])
ax.axhline((1 - df.label.mean()) * 100, ls='--', c='gray', lw=1, label=f"整体差评率 {100*(1-df.label.mean()):.1f}%")
for b, (n, r) in zip(bars, cat_stats.values):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
            f"{r*100:.1f}%\n(n={n})", ha='center', fontsize=9)
ax.set_ylabel('差评率 (%)'); ax.set_title('各品类差评率对比'); ax.legend()
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/cat_neg_rate.png', dpi=150); plt.close()

# ---- 2. 文本长度分布 ----
fig, ax = plt.subplots(figsize=(9, 5))
for lab, name, color in [(1, '好评', '#4a89dc'), (0, '差评', '#d64541')]:
    lens = df.loc[df.label == lab, 'len'].clip(upper=500)
    ax.hist(lens, bins=60, alpha=0.55, label=f"{name} (中位数 {df.loc[df.label==lab,'len'].median():.0f} 字)", color=color)
ax.set_xlabel('评论长度（字符，截断到500）'); ax.set_ylabel('条数')
ax.set_title('正/负评论文本长度分布'); ax.legend()
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/len_dist.png', dpi=150); plt.close()
summary['好评长度中位数'] = int(df.loc[df.label == 1, 'len'].median())
summary['差评长度中位数'] = int(df.loc[df.label == 0, 'len'].median())

# ---- 3. 正负评论高频词对比 ----
pos_words = pd.Series([w for t in df.loc[df.label == 1, 'review'] for w in tokens(t)]).value_counts()
neg_words = pd.Series([w for t in df.loc[df.label == 0, 'review'] for w in tokens(t)]).value_counts()
summary['好评Top10词'] = pos_words.head(10).to_dict()
summary['差评Top10词'] = neg_words.head(10).to_dict()
print('好评Top10:', dict(pos_words.head(10)))
print('差评Top10:', dict(neg_words.head(10)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, wc, title, color in [(axes[0], pos_words, '好评高频词 Top15', '#4a89dc'),
                             (axes[1], neg_words, '差评高频词 Top15', '#d64541')]:
    top = wc.head(15).iloc[::-1]
    ax.barh(top.index, top.values, color=color)
    ax.set_title(title)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/wordfreq.png', dpi=150); plt.close()

# ---- 4. 词云 ----
try:
    from wordcloud import WordCloud
    wc_cfg = dict(font_path='C:/Windows/Fonts/msyh.ttc', width=800, height=450,
                  background_color='white', max_words=120)
    for wc_series, name in [(pos_words, 'pos'), (neg_words, 'neg')]:
        WordCloud(colormap='Blues' if name == 'pos' else 'Reds', **wc_cfg)\
            .generate_from_frequencies(wc_series.to_dict())\
            .to_file(f'{FIG_DIR}/wordcloud_{name}.png')
except Exception as e:
    print('词云跳过:', e)

with open(f'{FIG_DIR}/eda_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print('EDA 完成，图已保存到', FIG_DIR)
