---
author: "Jorge Roldan"
date: '2025-02-27'
title: 'Evaluating Language Models'
tags: ['language_models', 'nlp', 'perplexity']
categories: ['notes']
ShowToc: true
ShowBreadCrumbs: false
draft: true
---

## Evaluating Language models: Training and Test sets {#evaluating-language-models-training-and-test-sets .unnumbered}

There are different ways of evaluating a language model such as
extrinsic, and intrinsic evaluation. In extrinsic evaluation we embed
the language model i an application and measure how the application's
performance improves, this is a very efficient way of evaluating models,
but it is unfortunately very expensive. On the other hand, a intrinsic
evaluation metric measures the quality of a model independent of an
application, one of this metrics is the **perplexity**.

### Types of datasets for model evaluation {#types-of-datasets-for-model-evaluation .unnumbered}

We need at least three types of datasets for evaluating a language
model: training, development, and test sets.

**Training set:** Dataset we use to learn the parameters of our model.

**Test set:** Separate dataset from the training set used to evaluate
the model. This test should reflect the language we want to use the
model for. After training two models in the training set, we can compare
how the two trained models fit the test set by determining which model
assigns a higher probability to the test set. We only want to test the
model once or very few times once on using the test set once the model
is ready.

**Development set:** We use the development set to do preliminary
testing and when we are ready we only use the test set once or very few
times.

## Evaluating Language models: Perplexity {#evaluating-language-models-perplexity .unnumbered}

**Perplexity**: The perplexity (PP, PPL) of a language model on a test
set is the inverse probability of the test set (one over the probability
of the test set), normalized by the number of words. This is why
sometimes it is called per-word perplexity.

For a test set $W=w_1 w_2 \dots w_N$:

$$\begin{aligned}
perplexity(W) &= P(w_1 w_2 \dots w_N)^{-\frac{1}{N}} \\ 
& = \sqrt[N]{\frac{1}{P(w_1 w_2 \dots w_N)}}
\end{aligned}$$

Using the chain rule we obtain:
$$perplexity(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i|w_1\dots w_{i-1})}}$$

Note that the higher the probability of a word sequence, the lower the
perplexity. Thus, the **lower the perplexity of a model on the data, the
better the model.**. Minimizing the perplexity is equivalent to
maximizing the test set probability according to the language model.

This is how we can calculate the perplexity of a unigram language model:

$$perplexity(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i)}}$$

And the same for a bigram model:

$$
\boxed{
perplexity(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i|w_{i-1})}}
}
$$