---
author: "Jorge Roldan"
date: '2025-02-22'
title: 'Using Mathjax for Latex support'
tags: ['latex', 'mathjax']
categories: ['tutorial']
ShowToc: true
ShowBreadCrumbs: true
# draft: true
---


## Introduction

This is **bold** text, and this is *emphasized* text.

Visit the [Hugo](https://gohugo.io) website!

$$
\begin{aligned} \label{eq1_test}
KL(\hat{y} || y) &= \sum_{c=1}^{M}\hat{y}_c \log{\frac{\hat{y}_c}{y_c}} \\
JS(\hat{y} || y) &= \frac{1}{2}(KL(y||\frac{y+\hat{y}}{2}) + KL(\hat{y}||\frac{y+\hat{y}}{2}))
\end{aligned} 
$$

This is equation \(\eqref{eq1_test}\)

$$
\begin{equation}
\end{equation}
$$


This is an inline \(a^*=x-b^*\) equation.

These are block equations:
\[a^*=x-b^*\]

\[ a^*=x-b^* \]

\[
a^*=x-b^*
\]

These are also block equations:

$$a^*=x-b^*$$

$$ a^*=x-b^* $$

$$
a^*=x-b^*
$$