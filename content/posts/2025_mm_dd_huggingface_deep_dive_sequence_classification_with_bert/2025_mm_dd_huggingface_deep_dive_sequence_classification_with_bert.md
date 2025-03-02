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
- [Transformer's pipeline](https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt)

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

Large language models have revolution



# Huggingface's transformers library
- Explain why I love transformers library
- Why is it important
- Model card

# Using a pretrained BERT model for sequence classification
## Model

Deep learning is a key AI technique Then [^1], [^2], [^3], [^4]
This is the start of my citations ,[[^aiayn]][[^bert]]


<!-- [^good_fellow]: Goodfellow, Ian, et al. *Deep Learning*. MIT Press, 2016. -->

# References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 

[^bert]: J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), J. Burstein, C. Doran, and T. Solorio, Eds., Minneapolis, Minnesota: Association for Computational Linguistics, Jun. 2019, pp. 4171–4186. doi: 10.18653/v1/N19-1423.

[^aiayn]: A. Vaswani et al., “Attention Is All You Need,” Aug. 01, 2023, arXiv: arXiv:1706.03762. Accessed: Apr. 03, 2024. [Online]. Available: http://arxiv.org/abs/1706.03762



[^1]: S. Iyer et al., “OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization,” Jan. 30, 2023, arXiv: arXiv:2212.12017. Accessed: Apr. 03, 2024. [Online]. Available: http://arxiv.org/abs/2212.12017

[^2]: D. G. Widder, S. West, and M. Whittaker, “Open (For Business): Big Tech, Concentrated Power, and the Political Economy of Open AI,” Aug. 17, 2023, Rochester, NY: 4543807. doi: 10.2139/ssrn.4543807.

[^3]: R. Bommasani et al., “On the Opportunities and Risks of Foundation Models,” Jul. 12, 2022, arXiv: arXiv:2108.07258. Accessed: Apr. 03, 2024. [Online]. Available: http://arxiv.org/abs/2108.07258

[^4]: F. Chollet, “On the Measure of Intelligence,” Nov. 25, 2019, arXiv:1911.01547. Accessed: Apr. 01, 2024. [Online]. Available: http://arxiv.org/abs/1911.01547

[^5]: V. Udandarao et al., “No ‘Zero-Shot’ Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance,” Apr. 04, 2024, arXiv:2404.04125. Accessed: Apr. 08, 2024. [Online]. Available: http://arxiv.org/abs/2404.04125

