---
author: "Jorge Roldan"
date: '2025-02-28'
title: 'Huggingface deep dive: Sequence Classification with BERT'
tags: ['sequence_classification', 'nlp', 'huggingface', 'transformers']
categories: ['tutorial']
ShowToc: true
ShowBreadCrumbs: true
draft: true
---

# Resources - tmp
- [lms_with_huggingface](https://github.com/roldanjrgl/lms_with_huggingface)
- [huggingface_deep_dive](https://github.com/roldanjrgl/huggingface_deep_dive/) 
- [sequence_classification.ipynb](https://github.com/roldanjrgl/huggingface_deep_dive/blob/main/sequence_classification.ipynb)
- [Huggingface's NLP-course](https://huggingface.co/learn/nlp-course/chapter1/1)

# TL;DR
```py
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

# Setup model and tokenizer
checkpoint = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = BertForSequenceClassification.from_pretrained(checkpoint)

# stage - 1
text = "I really love this book\n"
print(f"text:\t{text}")
inputs = tokenizer(text, return_tensors="pt")

# stage - 2
with torch.no_grad():
    logits = model(**inputs).logits
    
# stage - 3
predictions = torch.nn.functional.softmax(logits, dim=-1)
for id, label in model.config.id2label.items():
    print(f"{label:<7}:\t{round(float(predictions[0][id]), 3)}")
```



# Intro
- Why do we care about BERT?
- Why it's important to understand BERT in the age of Large Language Model?


# Huggingface's transformers library
- Explain why I love transformers library
- Why is it important
- Model card


# Using a pretrained BERT model for sequence classification
## Model
