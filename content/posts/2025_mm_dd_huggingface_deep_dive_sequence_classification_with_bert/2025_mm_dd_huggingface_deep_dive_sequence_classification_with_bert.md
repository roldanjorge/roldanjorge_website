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
- [BERT's Hugginface's model doc](https://huggingface.co/docs/transformers/model_doc/bert)


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

# Introduction
Large Language models (LLMs) have revolutionized Natural Language Processing (NLP) and are still transforming the field and its applications as of 2025. These models excel at common NLP tasks such as summarization, question answering, and text generation. A common trend in state-of-the-art LLMs is that they base their architecture on the Transformer's architecture [^aiayn], and decoder-only models have gained favorability compared to encoder-only or encoder-decoder models [^yang_harness].


In this article, I will discuss how to use the BERT model [^bert] for a sequence classification task with the [Huggingface's transformers library](https://huggingface.co/docs/transformers). So, why should we still care we care about BERT in 2025? First, its historical significance as one of the first models to showcase the power of the Transformer architecture, and anyone working with LLMs should be familiar with it. Second, smaller, encoder-only models such as BERT are better suited for powerful interpretability and explainability techniques, including LIME [^lime], SHAP [^shap], and attention visualization using tools such as BERTViz [^bertviz_paper],[^bertviz_repo] , or exBERT [^exbert] . Third, BERT models excel at tasks such as sequence classification, i.e., intent classification or sentiment analysis, and name entity recognition, and for specific applications, it is a better option than modern LLMs. Finally, BERT models are more cost-efficient, require fewer computing resources, are more environment-friendly, and can be easily deployed for large-scale applications than LLMs.

# Huggingface's transformers library
- Explain why I love transformers library
- Why is it important
- Model card

## Tokenizers

## Model checkpoints and architectures


# Using a pretrained BERT model for sequence classification

To easily run this code, please check [sequence_classification.ipynb](https://github.com/roldanjrgl/posts/blob/main/hf_deep_dive_seq_clas_with_bert/sequence_classification.ipynb). If you want to run it on your machine, just install the `transformers` package:

```bash
pip install transformers
```

For a detail guide on how to install packages on a conda environment, please check this article: [Setting up a Conda environment](https://www.roldanjorge.com/posts/2025_02_22_setting_up_a_conda_environment/setting_up_a_conda_environment/).

## Instantiate model and tokenizer
```py
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

# Setup model and tokenizer
checkpoint = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = BertForSequenceClassification.from_pretrained(checkpoint)
```

## Stage 1: Tokenize input
```py
# stage - 1
text = "I really love this book\n"
print(f"text:\t{text}")
inputs = tokenizer(text, return_tensors="pt")
```

## Stage 2: Model inference
```py
# stage - 2
with torch.no_grad():
    logits = model(**inputs).logits
```

## Stage 3: Post process results
```py
# stage - 3
predictions = torch.nn.functional.softmax(logits, dim=-1)
for id, label in model.config.id2label.items():
    print(f"{label:<7}:\t{round(float(predictions[0][id]), 3)}")
```


# References
[^bert]: J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), J. Burstein, C. Doran, and T. Solorio, Eds., Minneapolis, Minnesota: Association for Computational Linguistics, Jun. 2019, pp. 4171–4186. doi: 10.18653/v1/N19-1423.

[^aiayn]: A. Vaswani et al., “Attention Is All You Need,” Aug. 01, 2023, arXiv: arXiv:1706.03762. Accessed: Apr. 03, 2024. [Online]. Available: http://arxiv.org/abs/1706.03762


[^behind_the_pipeline]: “Behind the pipeline.” [Online]. Available: https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt


[^yang_harness]: J. Yang et al., “Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond,” Apr. 27, 2023, arXiv: arXiv:2304.13712. doi: 10.48550/arXiv.2304.13712.


[^lime]: M. T. Ribeiro, S. Singh, and C. Guestrin, “‘Why Should I Trust You?’: Explaining the Predictions of Any Classifier,” Aug. 09, 2016, arXiv: arXiv:1602.04938. doi: 10.48550/arXiv.1602.04938.

[^shap]: S. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” Nov. 25, 2017, arXiv: arXiv:1705.07874. doi: 10.48550/arXiv.1705.07874.


[^bertviz_paper]: J. Vig, “A multiscale visualization of attention in the transformer model,” in Proceedings of the 57th annual meeting of the association for computational linguistics: System demonstrations, M. R. Costa-jussà and E. Alfonseca, Eds., Florence, Italy: Association for Computational Linguistics, Jul. 2019, pp. 37–42. doi: 10.18653/v1/P19-3007.

[^bertviz_repo]: J. Vig, jessevig/bertviz. (Mar. 02, 2025). Python. Accessed: Mar. 02, 2025. [Online]. Available: https://github.com/jessevig/bertviz


[^exbert]: B. Hoover, bhoov/exbert. (Mar. 02, 2025). Python. Accessed: Mar. 02, 2025. [Online]. Available: https://github.com/bhoov/exbert
