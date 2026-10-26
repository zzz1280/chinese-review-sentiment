# -*- coding: utf-8 -*-
"""app.py — 中文购物评论情感分析演示（Gradio 网页）
用法:
    cd D:/my_projects/a/nlp-sentiment
    python app.py
然后浏览器打开 http://127.0.0.1:7860
优先加载微调好的 BERT 模型；若还没训练完，自动退回基线模型（TF-IDF+逻辑回归）。"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import gradio as gr

BERT_DIR = 'models/roberta-sentiment'

if os.path.isdir(BERT_DIR):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(DEVICE).eval()
    BACKEND = f'BERT (hfl/chinese-roberta-wwm-ext, {DEVICE})'
else:
    import joblib
    bundle = joblib.load('models/baseline.joblib')
    BACKEND = '基线模型 (TF-IDF + 逻辑回归)'

def predict(text):
    if os.path.isdir(BERT_DIR):
        with torch.no_grad():
            inputs = tok(text, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
            prob = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().tolist()
    else:
        prob = bundle['clf'].predict_proba(bundle['vectorizer'].transform([text]))[0].tolist()
    return {'差评 😠': round(prob[0], 4), '好评 😊': round(prob[1], 4)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label='输入商品评论', placeholder='例如：电池续航很差，半天就没电了'),
    outputs=gr.Label(label='情感预测', num_top_classes=2),
    examples=[
        ['质量非常好，物流也快，下次还来'],
        ['用了一周就坏了，客服还不理人'],
        ['味道不错，包装也很用心，就是有点贵'],
    ],
    title='中文购物评论情感分析',
    description=f'当前后端：{BACKEND}。训练数据：6.2 万条电商/酒店评论（10 个品类）。',
)

if __name__ == '__main__':
    demo.launch()
