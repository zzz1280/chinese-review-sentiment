# -*- coding: utf-8 -*-
"""02_baseline.py — 基线模型：TF-IDF + 逻辑回归，含跨领域测试（外卖评论）"""
import json
import os

import jieba
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

STOP = set("""
的 了 是 我 你 他 她 它 我们 你们 他们 也 都 就 不 很 人 一 在 有 和 与 及 为 这 那
之 还 被 把 让 给 但 而 等 又 或 啊 吧 呢 吗 呀 哦 嗯 哈 唉 非常 就是 觉得 感觉 还是
已经 可以 没有 什么 这个 那个 自己 东西 时候 因为 所以 但是 而且 如果 虽然 然后 只是
一直 特别 更加 有点 一些 一次 然后 还是 一个 不是 就是 就是说 真的 感到 而且 另外
""".split())

def analyzer(text):
    return [w for w in jieba.lcut(str(text)) if len(w) > 1 and w not in STOP and not w.isdigit()]

df = pd.read_csv('data/online_shopping_10_cats.csv').dropna(subset=['review'])
idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df['cat'])
print(f'训练集 {len(idx_train)} 条 / 测试集 {len(idx_test)} 条')

vec = TfidfVectorizer(analyzer=analyzer, max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = vec.fit_transform(df.loc[idx_train, 'review'])
X_test = vec.transform(df.loc[idx_test, 'review'])
y_train, y_test = df.loc[idx_train, 'label'], df.loc[idx_test, 'label']

clf = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

metrics = {
    'accuracy': round(float(accuracy_score(y_test, pred)), 4),
    'macro_f1': round(float(f1_score(y_test, pred, average='macro')), 4),
    'confusion_matrix': confusion_matrix(y_test, pred).tolist(),
    'classification_report': classification_report(y_test, pred, target_names=['差评', '好评'], output_dict=True),
}

# 分品类准确率
cat_test = df.loc[idx_test, 'cat']
per_cat = {}
for cat in cat_test.unique():
    m = cat_test == cat
    per_cat[cat] = round(float((pred[m.values] == y_test[m.values]).mean()), 4)
metrics['分品类准确率'] = per_cat

# 最容易误判的品类
worst = sorted(per_cat.items(), key=lambda kv: kv[1])[:3]
metrics['最易误判品类'] = dict(worst)

# 错分示例（模型自信错判的：差评判成好评）
proba = clf.predict_proba(X_test)[:, 1]
err = pd.DataFrame({
    'review': df.loc[idx_test, 'review'].values,
    '真实': y_test.values, '预测': pred, 'P(好评)': proba.round(3),
})
err = err[err['真实'] != err['预测']].nlargest(5, 'P(好评)')
metrics['高置信误判示例'] = err.to_dict('records')

# ---- 跨领域测试：直接用购物评论模型预测外卖评论 ----
waimai = pd.read_csv('data/waimai_10k.csv').dropna(subset=['review'])
w_pred = clf.predict(vec.transform(waimai['review']))
metrics['跨领域_waimai_accuracy'] = round(float((w_pred == waimai['label']).mean()), 4)

joblib.dump({'vectorizer': vec, 'clf': clf}, 'models/baseline.joblib')
with open('results/baseline_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

print('准确率:', metrics['accuracy'], '| macro-F1:', metrics['macro_f1'])
print('最易误判品类:', metrics['最易误判品类'])
print('跨领域(外卖)准确率:', metrics['跨领域_waimai_accuracy'])
print('完成')
