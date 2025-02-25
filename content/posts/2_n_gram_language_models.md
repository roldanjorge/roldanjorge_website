---
author: "Jorge Roldan"
date: '2025-02-22'
title: 'N-Gram Language Models'
tags: ['nlp', 'language_models']
categories: ['notes']
ShowToc: true
ShowBreadCrumbs: true
draft: true
---


# Equation 
$$
\boxed{
    P(X_1, \dots, X_n) = P(X_1) P(X_2 | X_1) P(X_3 | X_{1:2} \dots P(X_n | X_1:{n-1})
}
$$

$$
\begin{equation} P(X_1, \dots, X_n) = \prod_{k=1}^{n} P(X_k | X_{1:k-1})
\end{equation}
$$


We apply this to words:

\begin{equation}
    P(w_{1:n}) = P(w_1) P(w_2|w_1) P(w_3|w_{1:2}) \dots P(w_n|w_{1:n-1})
\end{equation}

\begin{equation}
   P(w_{1:n}) = \prod_{k=1}^{n} P(w_k|w_{1:k-1})
\end{equation}

It is still hard to compute $P(w_{1:n})$  using the chain rule. The key insight of the n-gram model is that we can approximate the history  just using the last few words. In the case of a bigram model, we approximate $P(w_n  | w_{1:n-1})$ by using only the probability of the preceding word $$P(w_n|w_{n-1})$$

\begin{equation}
    P(w_n|w_{1:n-1}) \approx P(w_n|w_{n-1})
\end{equation}

An example of a \textbf{Markov assumption} is when the probability of a word depends only on the previous word in the case of a bigram, a trigram, and a n-gram looks two words, and $n-1$ 
words into the past, respectively. 

Generalization:

\begin{equation}
   P(w_n|w_{1:n-1}) \approx P(w_n|w_{n-N+1:n-1}) 
\end{equation}



In the case of the bigram, we have:

\begin{equation}
\label{bigram_aprox}
    P(w_{1:n}) = \prod_{k=1}^{n} P(w_k|w_{1:k-1}) \approx  \prod_{k=1}^{n} P(w_k|w_{k-1})
\end{equation}

We can estimate  equation\ref{bigram_aprox} using the MLE (Maximum likelihood estimation) 

\begin{equation}
P(w_n | w_{n-1}) = \frac{C(w_{n-1}w_n)}{\sum_{w} C(w_{n-1} w)}
\end{equation}
Which can be simplified to

\begin{equation}
P(w_n | w_{n-1}) = \frac{C(w_{n-1}w_n)}{C(w_{n-1} )}
\end{equation}

We can generalize the estimation of the MLE n-gram parameter as follows:

\begin{equation}
    P(w_n|w_{n-N+1:n-1}) = \frac{C(w_{n-N+1:n-1} \  w_n)}{C(w_{n-N+1:n-1})}
\end{equation}
This ratio is called the **relative frequency**


# Evaluating Language models: Training and Test sets

There are different ways of evaluating a language model such as extrinsic, and intrinsic evaluation. In extrinsic evaluation we embed the language model i an application and measure how the application's performance improves, this is a very efficient way of evaluating models, but it is unfortunately very expensive. On the other hand, a intrinsic evaluation metric measures the quality of a model independent of an application, one of this metrics is the \textbf{perplexity}.

### Types of datasets for model evaluation
We need at least three types of datasets for evaluating a language model: training, development, and test sets. 

\begin{tcolorbox} 
[colback=blue!5!white,colframe=blue!75!black,title=Types of datasets for model evaluation]

\textbf{Training set:} \newline
Dataset we use to learn the parameters of our model.

\textbf{Test set:} \newpage
Separate dataset from the training set used to evaluate the model. This test should reflect the language we want to use the model for.  After training two models in the training set, we can compare how the two trained models fit the test set by determining which model assigns a higher probability to the test set.  We only want to test the model once or very few times once on using the test set once the model is ready.

\textbf{Development set:} \newpage
We use the development set to do preliminary testing and when we are ready we only use the test set once or very few times.
\end{tcolorbox}


## 3.3. Evaluating Language models: Perplexity



\begin{tcolorbox} 
[colback=blue!5!white,colframe=blue!75!black,title=Perplexity]
\textbf{Perplexity}:  The perplexity (PP, PPL) of a language model on a test set is the inverse probability of the test set (one over the probability of the test set), normalized by the number of words. This is why sometimes it is called per-word perplexity.

For a test set $W=w_1 w_2 \dots w_N$:

\begin{align}
perplexity(W) &= P(w_1 w_2 \dots w_N)^{-\frac{1}{N}} \\ 
& = \sqrt[N]{\frac{1}{P(w_1 w_2 \dots w_N)}}
\end{align}

Using the chain rule we obtain:
 \begin{equation}
perplexity(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i|w_1\dots w_{i-1})}}
\end{equation}

\end{tcolorbox}
Note that the higher the probability of a word sequence, the lower the perplexity. Thus, the \textbf{lower the perplexity of a model on the data, the better the model.}. Minimizing the perplexity is equivalent to maximizing the test set probability according to the language model.

This is how we can calculate the perplexity of a unigram language model:

\begin{equation}
   perplexity(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i)}} 
\end{equation}

And the same for a bigram model:
\begin{equation}
   perplexity(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i|w_{i-1})}} 
\end{equation}

## 3.4. Sampling sentences from a language model
One technique to visualize the knowledge of a model is to sample from it. Sampling from a distribution means to choose random points according to their likelihood. Sampling from a language model, which represents a distribution over sentences, means to generate some sentences, choosing each sentence according to its likelihood as defined by the model. 


