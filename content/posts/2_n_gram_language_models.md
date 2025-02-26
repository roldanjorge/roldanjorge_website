---
author: "Jorge Roldan"
date: '2025-02-26'
title: 'N-Gram Language Models'
tags: ['ngram', 'language_models', 'nlp']
categories: ['notes']
ShowToc: true
ShowBreadCrumbs: false
---

# N-gram Language Models {#n-gram-language-models}

## N-Grams
N-gram models are the simplest type of language models. The N-gram term
has two meanings. One meaning refers to a sequence of n words, so a
2-gram, and 3-gram are sequences of 2, and 3 words, respectively. The
second meaning refers to a probabilistic model that estimates the
probability of a word given the n-1 previous words.

We represent a sequence of n words as $w_1 \dots w_n$ or $w_{1:n}$ , and
the join probability of each word in a sequence having a value:
$P(W_1 = w_1, X_2=w_2, X_3=w_3, \dots, X_n = w_n$

To compute the probability of $P(w_1, w_2, \dots, w_n)$ we use the chain
rule of probability

$$P(X_1, \dots, X_n) = P(X_1) P(X_2 | X_1) P(X_3 | X_{1:2} \dots P(X_n | X_1:{n-1})$$
$$P(X_1, \dots, X_n) = \prod_{k=1}^{n} P(X_k | X_{1:k-1})$$

We apply this to words:

$$P(w_{1:n}) = P(w_1) P(w_2|w_1) P(w_3|w_{1:2}) \dots P(w_n|w_{1:n-1})$$

$$P(w_{1:n}) = \prod_{k=1}^{n} P(w_k|w_{1:k-1})$$

It is still hard to compute $P(w_{1:n})$ using the chain rule. The key
insight of the n-gram model is that we can approximate the history just
using the last few words. In the case of a bigram model, we approximate
$P(w_n  | w_{1:n-1})$ by using only the probability of the preceding
word $P(w_n|w_{n-1})$

$$P(w_n|w_{1:n-1}) \approx P(w_n|w_{n-1})$$ An example of a **Markov
assumption** is when the probability of a word depends only on the
previous word in the case of a bigram, a trigram, and a n-gram looks two
words, and $n-1$ words into the past, respectively.

Generalization:

$$P(w_n|w_{1:n-1}) \approx P(w_n|w_{n-N+1:n-1})$$

In the case of the bigram, we have:

$$\label{bigram_aprox}
    P(w_{1:n}) = \prod_{k=1}^{n} P(w_k|w_{1:k-1}) \approx  \prod_{k=1}^{n} P(w_k|w_{k-1})$$

We can estimate
equation[\[bigram_aprox\]](#bigram_aprox){reference-type="ref"
reference="bigram_aprox"} using the MLE (Maximum likelihood estimation)

$$P(w_n | w_{n-1}) = \frac{C(w_{n-1}w_n)}{\sum_{w} C(w_{n-1} w)}$$ Which
can be simplified to

$$P(w_n | w_{n-1}) = \frac{C(w_{n-1}w_n)}{C(w_{n-1} )}$$

We can generalize the estimation of the MLE n-gram parameter as follows:

$$P(w_n|w_{n-N+1:n-1}) = \frac{C(w_{n-N+1:n-1} \  w_n)}{C(w_{n-N+1:n-1})}$$
This ratio is called the **relative frequency**

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

## Sampling sentences from a language model {#sampling-sentences-from-a-language-model .unnumbered}

One technique to visualize the knowledge of a model is to sample from
it. Sampling from a distribution means to choose random points according
to their likelihood. Sampling from a language model, which represents a
distribution over sentences, means to generate some sentences, choosing
each sentence according to its likelihood as defined by the model.
