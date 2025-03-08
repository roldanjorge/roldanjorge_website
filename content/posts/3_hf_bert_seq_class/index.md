---
author: "Jorge Roldan"
date: '2025-03-08'
title: 'Huggingface deep dive: Sequence Classification with BERT'
tags: ['sequence_classification', 'nlp', 'huggingface', 'transformers']
categories: ['tutorials']
ShowToc: true
ShowBreadCrumbs: true
---

# Introduction
Large Language models (LLMs) have revolutionized Natural Language Processing (NLP) and are still transforming the field and its applications as of 2025. These models excel at common NLP tasks such as summarization, question answering, and text generation. A common trend in state-of-the-art LLMs is that they base their architecture on the Transformer's architecture [^aiayn], and decoder-only models have gained favorability compared to encoder-only or encoder-decoder models [^yang_harness].


In this article, I will discuss how to use the BERT model [^bert] for a sequence classification task with the [Huggingface's transformers library](https://huggingface.co/docs/transformers). So, why should we still care we care about BERT in 2025? First, its historical significance as one of the first models to showcase the power of the Transformer architecture, and anyone working with LLMs should be familiar with it. Second, smaller, encoder-only models such as BERT are better suited for powerful interpretability and explainability techniques, including LIME [^lime], SHAP [^shap], and attention visualization using tools such as BERTViz [^bertviz_paper],[^bertviz_repo] , or exBERT [^exbert] . Third, BERT models excel at tasks such as sequence classification, i.e., intent classification or sentiment analysis, and name entity recognition, and for specific applications, it is a better option than modern LLMs. Fourth, BERT models are more cost-efficient, require fewer computing resources, are more environment-friendly, and can be more easily deployed for large-scale applications than LLMs. Finally, if you learn how to use BERT with the `transformers` library, you can apply the same skills to other state-of-the-art open-source LLMs. 

# Huggingface's transformers library
Huggingface's `transformers` is a wonderful open-source library that empowers users to use pre-trained models for multiple tasks in modalities such as Natural Language Processing, Computer Vision, Audio, and Multimodel. One of its core advantages is its support and interoperability between various frameworks such as PyTorch, TensorFlow, and JAX. [^transformers_doc] . You can find a list of the models supported here [Supported models and frameworks](https://huggingface.co/docs/transformers/index#supported-models-and-frameworks), and comprehensive documentation for [BERT](https://huggingface.co/docs/transformers/model_doc/bert) [^bert_hf_docs]


## Model checkpoints and architectures {#model_checkpoints_and_architectures}
Using BERT requires choosing an architecture and a checkpoint. A checkpoint indicates the state of a pre-trained model, such as its weights and configuration. These are some examples of widely used BERT checkpoints.


| Checkpoint (model-card)                                                                      | Notes                                         |
| -------------------------------------------------------------------------------------------- | --------------------------------------------- |
| [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                | Trained on lowercased English text            |
| [bert-large-uncased](https://huggingface.co/bert-large-uncased)                              | Larger version of bert-base-uncased           |
| [bert-base-cased](https://huggingface.co/google-bert/bert-base-uncased)                      | Account for capitalization                    |
| [bert-large-cased](https://huggingface.co/bert-large-cased)                                  | Larger version of bert-base-case              |
| [bert-base-multilingual-uncased-sentiment](nlptown/bert-base-multilingual-uncased-sentiment) | Finetuned for sentiment analysis              |
| [bert-base-ner](https://huggingface.co/dslim/bert-base-NER)                                  | Fine-tuned for Named Entity Recognition (NER) |


The choice of architecture depends on the task that you are planning to do. These are some of the main architetures used with BERT.


| Task                    | Architecture                                                                                                                                   |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Sequence Classification | [BertForSequenceClassification](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/bert#transformers.BertForSequenceClassification) |
| Token Classification    | [BertForTokenClassification](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/bert#transformers.BertForTokenClassification)       |
| Fill Mask               | [BertForMaskedLM](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/bert#transformers.BertForMaskedLM)                             |
| Question Answering      | [BertForQuestionAnswering](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/bert#transformers.BertForQuestionAnswering)           |
| Multiple choice         | [BertForMultipleChoice](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/bert#transformers.BertForMultipleChoice)                 |


# Using a pretrained BERT model for sequence classification
## Pipeline overview

The three stages for sequence classification with BERT are as follows: stage 1: preprocessing, where we convert the utterance into tensors using the tokenizer. Next, in stage 2, we use these tensors as inputs for the model, and the model outputs logits. Finally, these logits are converted into probabilities using the Softmax function.

{{< figure id="pipeline_hl" src="./pipeline_high_level.png" alt="Sample figure" caption="High-level stages for sequence classification with BERT" >}}

{{< figref "pipeline_hl" >}} illustrated these three stages at a high-level. We will implement each stage and discuss the results in later sections.


## Complete source code {#complete_source_code}

To efficiently run this code, please check [sequence_classification.ipynb](https://github.com/roldanjorge/posts/blob/main/hf_bert_seq_class/sequence_classification.ipynb) or [sequence_classification.py](https://github.com/roldanjorge/posts/blob/main/hf_bert_seq_class/sequence_classification.py). If you want to run it on your machine, install the [transformers](https://huggingface.co/docs/transformers/en/installation) and [torch](https://pytorch.org/get-started/locally/#linux-pip) packages.

For a detailed guide on how to install packages on a Conda environment, please check this article: [Setting up a Conda environment](https://www.roldanjorge.com/posts/1_setting_up_a_conda_env/index).

<details>
  <summary>Show Code</summary>

```py
"""
This script demonstrates the pipeline for sequence classification using Huggingface transformers.
"""
import os
import torch
from typing import List
from transformers import AutoTokenizer, BertForSequenceClassification


def get_model_tokenizer(checkpoint: str, output_dir: str) -> (AutoTokenizer, BertForSequenceClassification):
    """ Download or load from local and return the model and its tokenizer

    Args:
        checkpoint: Huggingface checkpoint
        output_dir: Directory to store model and tokenizer file

    Returns:
        tokenizer: Tokenizer object
        model: Model object
    """
    os.makedirs(output_dir, exist_ok=True)

    # Download model and tokenizer
    model = BertForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Store model and tokenizer in output_dir
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return tokenizer, model


def get_id_token_mapping(inputs, tokenizer) -> List[dict]:
    """ Get the mapping between the token id and its respective token

        Args:
            inputs: Output of tokenizer containing input_ids
            tokenizer: The tokenizer object
    """
    _mapping = []
    id2token = {value: str(key) for key, value in tokenizer.vocab.items()}
    input_ids = inputs.input_ids[0].tolist()
    for token_id in input_ids:
        _mapping.append({str(token_id): id2token.get(token_id)})
    return _mapping


def run_pipeline(utterance: str, tokenizer, model: BertForSequenceClassification):
    """ Run the pipeline for the sequence classification task
        Args:
            utterance: Input text
            tokenizer: Tokenizer object
            model: Model object
    """
    print(f"\n{50 * '='}\nRunning pipeline: \"{utterance}\"\n{50 * '='}")

    # Stage 1: Preprocessing
    print(f"{50 * '-'}\nStage 1: Preprocessing \n{50 * '-'}")
    inputs = tokenizer(utterance, return_tensors="pt")
    for _input, value in inputs.items():
        print(f"{_input:<15}: \n\t{value}")

    print(f"\n** Additional details (token_id to token mapping) **")
    mapping = get_id_token_mapping(inputs=inputs, tokenizer=tokenizer)
    print(f"mapping: \n\t{mapping}")

    # Stage 2: Model inference
    print(f"\n{50 * '-'}\nStage 2: Model inference \n{50 * '-'}")
    with torch.no_grad():
        logits = model(**inputs).logits
    print(f"logits: \n\t{logits}")

    # Stage 3: Post-processing
    print(f"\n{50 * '-'}\nStage 3: Post-processing \n{50 * '-'}")
    predictions = torch.nn.functional.softmax(logits, dim=-1)
    print(f"probabilities: \n\t{predictions}")
    print(f"id2label: \n\t{model.config.id2label}")
    print(f"predictions:")
    for _id, label in model.config.id2label.items():
        print(f"\t{label:<7}:\t{round(float(predictions[0][_id]), 3)}")


def main():
    # Setup tokenizer and model
    checkpoint = "nlptown/bert-base-multilingual-uncased-sentiment"
    output_dir = 'model'
    tokenizer, model = get_model_tokenizer(checkpoint=checkpoint, output_dir=output_dir)

    # Positive review
    run_pipeline(utterance="I really loved that movie", tokenizer=tokenizer, model=model)

    # Negative review
    run_pipeline(utterance="I hate very cold, and cloudy winter days", tokenizer=tokenizer, model=model)


if __name__ == "__main__":
    main()
```
</details>

## Instantiate model and tokenizer
Note: Complete source code is included here [complete code](#complete_source_code)


### Downloading and storing the model and tokenizer
How do we download a Hugginface's model and its respective tokenizer? We only need a checkpoint and its respective architecture, as mentioned in [here](#model_checkpoints_and_architectures). For this post, we will be using the checkpoint [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) to do sentiment analysis for product review, and the [BertForSequenceClassification](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/bert#transformers.BertForSequenceClassification) architecture. Remember that we use the AutoTokenizer class to download the correct tokenizer automatically using the checkpoint. If you run the code, you will see that the tokenizer is a [BertTokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertTokenizerFast) object with a vocabulary size of **105,879**.


```py
import os
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

checkpoint = "nlptown/bert-base-multilingual-uncased-sentiment"
output_dir = 'model'

# Download model and tokenizer
model = BertForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Store model and tokenizer in output_dir
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```


{{< figure id="model_files" src="./model_files.png" alt="Sample figure" caption="Model and tokenizer files" width="330" height="160" >}}

If you run the above script, you will see in {{< figref "model_files" >}} that the model and tokenizer files are stored in a `model` directory. The `config.json` has the core information, such as model name, architecture, and output details. Also,  the model weights are stored in the `model.safetensor`. Please review these files to understand the model we will use better.


<details>
  <summary>config.json</summary>

```json
{
  "_name_or_path": "nlptown/bert-base-multilingual-uncased-sentiment",
  "_num_labels": 5,
  "architectures": [
    "BertForSequenceClassification"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "finetuning_task": "sentiment-analysis",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "1 star",
    "1": "2 stars",
    "2": "3 stars",
    "3": "4 stars",
    "4": "5 stars"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "1 star": 0,
    "2 stars": 1,
    "3 stars": 2,
    "4 stars": 3,
    "5 stars": 4
  },
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "torch_dtype": "float32",
  "transformers_version": "4.49.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 105879
}
```
</details>


## Stage 1: Tokenize input
Note: Complete source code is included here [complete code](#complete_source_code)

Transformer-based models such as BERT cannot process raw utterances. We first need to use the tokenizer to convert a string into multiple tensors, which will be the actual inputs to the model as illustrated in {{< figref "pipeline_hl" >}}. These tensors are `inputs_ids`, `token_type_ids`, and the `attention_mask`.

- `inputs_ids`: Represents each token with an id based on the model's vocabulary.
- `token_type_ids`: Indicates which tokens should be attended to or ignored using 1 or 0, respectively. 
- `attention_mask`: Distinguishes segments in an input, where each integer belongs to one specific segment. For inputs with one segment, all values will be 0.


```py
 # Stage 1: Preprocessing
print(f"{50 * '-'}\nStage 1: Preprocessing \n{50 * '-'}")
inputs = tokenizer(utterance, return_tensors="pt")
for _input, value in inputs.items():
print(f"{_input:<15}: \n\t{value}")

print(f"\n** Additional details (token_id to token mapping) **")
mapping = get_id_token_mapping(inputs=inputs, tokenizer=tokenizer)
print(f"mapping: \n\t{mapping}")
```

This is the output when using the positive review: `"I really loved that movie"`. Please note the token_id to token mapping at the end. This shows how the tokenizer splitted the utterance into tokens, and converted that token into a token id. Also note that `{'101': '[CLS]'}`, and `{'102': '[SEP]'}` are special tokens, corresponding to the classification, and separation tokens, respectively.

<details>
<summary>output</summary>

```bash
==================================================
Running pipeline: "I really loved that movie"
==================================================
--------------------------------------------------
Stage 1: Preprocessing 
--------------------------------------------------
input_ids      : 
        tensor([[  101,   151, 25165, 46747, 10203, 13113,   102]])
token_type_ids : 
        tensor([[0, 0, 0, 0, 0, 0, 0]])
attention_mask : 
        tensor([[1, 1, 1, 1, 1, 1, 1]])

** Additional details (token_id to token mapping) **
mapping: 
        [{'101': '[CLS]'}, {'151': 'i'}, {'25165': 'really'}, {'46747': 'loved'}, {'10203': 'that'}, {'13113': 'movie'}, {'102': '[SEP]'}]
```
</details>

## Stage 2: Model inference
Note: Complete source code is included here [complete code](#complete_source_code)

Model inference is the stage where we use the model weights to get a result. In this case, we want to predict the rating (star (s) from 1 to 5) based on a user utterance. 

```py
# Stage 2: Model inference
print(f"\n{50 * '-'}\nStage 2: Model inference \n{50 * '-'}")
with torch.no_grad():
logits = model(**inputs).logits
print(f"logits: \n\t{logits}")
```

<details>

<summary>output</summary>

```bash
--------------------------------------------------
Stage 2: Model inference 
--------------------------------------------------
logits: 
        tensor([[-2.3669, -2.2634, -0.4449,  1.5619,  2.7230]])
```
</details>

## Stage 3: Post process results
Note: Complete source code is included here [complete code](#complete_source_code) 

The raw output of the model in stage 2 is called logits. We need to apply the [softmax fuction](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch-nn-functional-softmax) to get the probability distribution over the 5 stars ratings. Softmax in the context of PyTorch is defined as

$$ 
\begin{equation}
Softmax(x_i) = \frac{exp(x_i)}{\sum_j exp(x_j)}
\label{softmax}
\end{equation}
$$

Where Softmax  \eqref{softmax} is applied to all the slices along dim, in our case, `dim=-1 (5)` since we have 5 stars (1 -> 5).

```py
# Stage 3: Post-processing
print(f"\n{50 * '-'}\nStage 3: Post-processing \n{50 * '-'}")
predictions = torch.nn.functional.softmax(logits, dim=-1)
print(f"probabilities: \n\t{predictions}")
print(f"id2label: \n\t{model.config.id2label}")
print(f"predictions:")
for _id, label in model.config.id2label.items():
print(f"\t{label:<7}:\t{round(float(predictions[0][_id]), 3)}")
```


<details>

<summary>output</summary>

```bash
--------------------------------------------------
Stage 3: Post-processing 
--------------------------------------------------
probabilities: 
        tensor([[0.0045, 0.0050, 0.0308, 0.2289, 0.7309]])
id2label: 
        {0: '1 star', 1: '2 stars', 2: '3 stars', 3: '4 stars', 4: '5 stars'}
predictions:
        1 star :        0.005
        2 stars:        0.005
        3 stars:        0.031
        4 stars:        0.229
        5 stars:        0.731
```
</details>

## Examples
### Output: Positive review 
Complete pipeline's output when using a positive review such as "I really loved that movie".

{{< figure id="pipeline_positive_review" src="./pipeline_positive_review.png" alt="pipeline_positive_review" caption="Negative review example" >}}


```bash
==================================================
Running pipeline: "I really loved that movie"
==================================================
--------------------------------------------------
Stage 1: Preprocessing 
--------------------------------------------------
input_ids      : 
        tensor([[  101,   151, 25165, 46747, 10203, 13113,   102]])
token_type_ids : 
        tensor([[0, 0, 0, 0, 0, 0, 0]])
attention_mask : 
        tensor([[1, 1, 1, 1, 1, 1, 1]])

** Additional details (token_id to token mapping) **
mapping: 
        [{'101': '[CLS]'}, {'151': 'i'}, {'25165': 'really'}, {'46747': 'loved'}, {'10203': 'that'}, {'13113': 'movie'}, {'102': '[SEP]'}]

--------------------------------------------------
Stage 2: Model inference 
--------------------------------------------------
logits: 
        tensor([[-2.3669, -2.2634, -0.4449,  1.5619,  2.7230]])

--------------------------------------------------
Stage 3: Post-processing 
--------------------------------------------------
probabilities: 
        tensor([[0.0045, 0.0050, 0.0308, 0.2289, 0.7309]])
id2label: 
        {0: '1 star', 1: '2 stars', 2: '3 stars', 3: '4 stars', 4: '5 stars'}
predictions:
        1 star :        0.005
        2 stars:        0.005
        3 stars:        0.031
        4 stars:        0.229
        5 stars:        0.731
```

### Output: Negative review 

Complete pipeline's output when using a positive review such as "I really loved that movie".


{{< figure id="pipeline_negative_review" src="./pipeline_negative_review.png" alt="pipeline_negative_review" caption="Negative review example" >}}

```bash
==================================================
Running pipeline: "I hate very cold, and cloudy winter days"
==================================================
--------------------------------------------------
Stage 1: Preprocessing 
--------------------------------------------------
input_ids      : 
        tensor([[  101,   151, 39487, 12495, 19443,   117, 10110, 28419, 10158, 14690,
         12889,   102]])
token_type_ids : 
        tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
attention_mask : 
        tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

** Additional details (token_id to token mapping) **
mapping: 
        [{'101': '[CLS]'}, {'151': 'i'}, {'39487': 'hate'}, {'12495': 'very'}, {'19443': 'cold'}, {'117': ','}, {'10110': 'and'}, {'28419': 'cloud'}, {'10158': '##y'}, {'14690': 'winter'}, {'12889': 'days'}, {'102': '[SEP]'}]

--------------------------------------------------
Stage 2: Model inference 
--------------------------------------------------
logits: 
        tensor([[ 0.7603,  0.8743, -0.0698, -0.7666, -0.7647]])

--------------------------------------------------
Stage 3: Post-processing 
--------------------------------------------------
probabilities: 
        tensor([[0.3343, 0.3746, 0.1457, 0.0726, 0.0727]])
id2label: 
        {0: '1 star', 1: '2 stars', 2: '3 stars', 3: '4 stars', 4: '5 stars'}
predictions:
        1 star :        0.334
        2 stars:        0.375
        3 stars:        0.146
        4 stars:        0.073
        5 stars:        0.073
```

# Conclusion 
You now have all the tools to use BERT for sequence classification. Please check the vast number of checkpoints and architectures you could use for various applications. You can see some of the most common ones [here](#model_checkpoints_and_architectures). Furthermore, if you want to better understand this material, one great way to do it is to run the scripts provided with a debugger and see the results yourself. Finally, remember that you can apply these skills to download and start using some of the state-of-the-art LLM models listed [here](https://huggingface.co/docs/transformers/index#supported-models-and-frameworks).

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


[^transformers_doc]: “🤗 Transformers.” Accessed: Mar. 03, 2025. [Online]. Available: https://huggingface.co/docs/transformers/index


[^bert_hf_docs]: “BERT.” Accessed: Mar. 03, 2025. [Online]. Available: https://huggingface.co/docs/transformers/model_doc/bert
