---
author: "Jorge Roldan"
date: '2025-02-27'
title: 'N-Gram Language Models'
tags: ['ngram', 'language_models', 'nlp']
categories: ['notes']
ShowToc: true
ShowBreadCrumbs: false
cover:
  image: "images/posts/2025_02_27_ngram_lms/image.png"
---

This post is based chapter 3 from [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) by Dan Jurafsky and James H. Martin

# N-Grams models
N-gram models are the simplest type of language models. The N-gram term
has two meanings. One meaning refers to a sequence of n words, so a
2-gram, and 3-gram are sequences of 2, and 3 words, respectively. The
second meaning refers to a probabilistic model that estimates the
probability of a word given the n-1 previous words.

We represent a sequence of n words as $w_1 \dots w_n$ or $w_{1:n}$ , and
the join probability of each word in a sequence having a value:
$P(W_1 = w_1, X_2=w_2, X_3=w_3, \dots, X_n = w_n$) as $P(w_1, w_2, \dots, w_n)$

## Chain rule
To compute the probability of $P(w_1, w_2, \dots, w_n)$ we use the [chain
rule of probability](https://en.wikipedia.org/wiki/Chain_rule_(probability))

$$
\begin{equation}
\label{chain_rule1}
P(X_1, \dots, X_n) = P(X_1) P(X_2 | X_1) P(X_3 | X_{1:2}) \dots P(X_n | X_{1:{n-1}})
\end{equation}
$$
$$
\begin{equation}
\label{2}
P(X_1, \dots, X_n) = \prod_{k=1}^{n} P(X_k | X_{1:k-1})
\end{equation}
$$

## Applying chain rule to a sequence of words
We apply the chain rule \eqref{chain_rule1}, \eqref{2} to a sequence of words to get the following

$$P(w_{1:n}) = P(w_1) P(w_2|w_1) P(w_3|w_{1:2}) \dots P(w_n|w_{1:n-1})$$

$$P(w_{1:n}) = \prod_{k=1}^{n} P(w_k|w_{1:k-1})$$

It is still hard to compute $P(w_{1:n})$  using the chain rule. The key
insight of the n-gram model is that we can approximate the history just
using the last few words. In the case of a bigram model, we approximate
$P(w_n  | w_{1:n-1})$ by using only the probability of the preceding
word $P(w_n|w_{n-1})$

$$P(w_n|w_{1:n-1}) \approx P(w_n|w_{n-1})$$ 

## Markov Assumptions
The key insight of the n-gram model is based on a **Markov
assumption**. We make a Markov assumption when we assume that the probability of a word depends only on the
previous word in the case of a bigram, the last two words in the case of a trigram, and $n-1$ words in the case of a n-gram.

We can generalize the Markov assumption this way:

$$
\boxed{
P(w_n|w_{1:n-1}) \approx P(w_n|w_{n-N+1:n-1})
}
$$

In the case of the bigram, we have:

$$\begin{equation}
\boxed{
    \label{bigram_aprox}
    P(w_{1:n}) = \prod_{k=1}^{n} P(w_k|w_{1:k-1}) \approx  \prod_{k=1}^{n} P(w_k|w_{k-1})
}
\end{equation}
$$

## Estimating a bigram model using MLE (Maximum Likelihood Estimation)
We can estimate equation \eqref{bigram_aprox} using the MLE (Maximum likelihood estimation)

$$P(w_n | w_{n-1}) = \frac{C(w_{n-1}w_n)}{\sum_{w} C(w_{n-1} w)}$$ Which
can be simplified to

$$P(w_n | w_{n-1}) = \frac{C(w_{n-1}w_n)}{C(w_{n-1} )}$$

We can generalize the estimation of the MLE n-gram parameter as follows:

$$
\boxed{
    P(w_n|w_{n-N+1:n-1}) = \frac{C(w_{n-N+1:n-1} \  w_n)}{C(w_{n-N+1:n-1})}
}
$$
This ratio is called the **relative frequency**


# References
[1] Daniel Jurafsky and James H. Martin. 2025. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models, 3rd edition. Online manuscript released January 12, 2025. https://web.stanford.edu/~jurafsky/slp3.