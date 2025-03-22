---
author: "Jorge Roldan"
date: '2025-03-08'
title: 'Regularization'
tags: ['regularization']
categories: ['notes']
ShowToc: true
ShowBreadCrumbs: true
---

## Intro to Regularization and why it is important

Linear regression is a foundational model in Statistics and Machine Learning. However, it has

## Linear regression models  and Least squares
Linear regression is represented by equation $\ref{lin_reg_eq}$ where $x^T = (x_1, x_2, \dots, x_p)$ is a vector with dimension p,  $y$ is a real-valued output, and the values for $\beta_j$'s  are the unknown parameters of the model. [1]

$$
\begin{equation} 
\label{lin_reg_eq}
f(x) = \beta_0 + \sum^{p}_{j = 1} x_j \beta_j
\end{equation}
$$

The least squares method is commonly used to determine the  parameters $\beta_j$'s  by minimizing the difference $y_i - f(x_i)$ for each of the points in the training dataset $(x_1, y_1), \dots, (x_N, y_N)$ as in equation $\eqref{rss_1}$, where RSS stands for residual sum-of-squares.

$$
\begin{equation}
\label{rss_1}
RSS(\beta) = \sum_{i=1}^{N} (y_i - f(x_i))^2
\end{equation}
$$

Substituting $\ref{lin_reg_eq}$ into $\ref{rss_1}$ we obtain $\ref{rss_2}$

$$
\begin{equation}
\label{rss_2}
RSS(\beta) = \sum_{i=1}^{N} (y_i - \beta_0 - \sum^{p}_{j = 1} X_j \beta_j )^2
\end{equation}
$$



We can represent  equation $\ref{rss_2}$  in matrix form by using all the samples in our training set at once. We stack all the $N$ inputs of $p+1$-dimensional $x_i$ vectors in our training data to create $\pmb{X}$, all the  $N$  $y_i$ outputs into  a $\pmb{y}$ vector, and the  $p + 1$ parameters into $\beta$ . Note that  the  first element $x_{i1}$  is  always a 1 which multiplies with $\beta_0$.  The matrix equation are shown in equation $\ref{rss_matrix}$ and the extended version on $\ref{rss_matrix_ext}$

$$
\begin{equation}
\label{rss_matrix}
RSS(\beta) = (\pmb{y}  -  \pmb{X}\beta)^T(\pmb{y} - \pmb{X}\beta)
\end{equation}
$$

{{< rawhtml >}}
$$
\begin{equation}
\label{rss_matrix_ext}
RSS(\beta) =
\left(\left[ {\begin{array}{cc}
y_{1}  \\
\vdots \\
y_{N}  \\
\end{array} } \right] 
- 

\left[ {\begin{array}{cc}
x_{11} & \cdots & x_{1(p+1)} \\
\vdots \\
x_{N1} & \cdots & x_{N(p+1)} \\
\end{array} } \right]

\left[ {\begin{array}{cc}
\beta_{0}  \\
\vdots \\
\beta_{p}  \\
\end{array} } \right] 
\right)^T
\left(\left[ {\begin{array}{cc}
y_{1}  \\
\vdots \\
y_{N}  \\
\end{array} } \right] 
- 

\left[ {\begin{array}{cc}
x_{11} & \cdots & x_{1(p+1)} \\
\vdots \\
x_{N1} & \cdots & x_{N(p+1)} \\
\end{array} } \right]

\left[ {\begin{array}{cc}
\beta_{0}  \\
\vdots \\
\beta_{p}  \\
\end{array} } \right] 
\right)
\end{equation}
$$
{{< /rawhtml >}}


Now that we have equation $\eqref{rss_matrix}$ in terms of $\pmb{X}$, $\pmb{y}$, and $\beta$. To find the value of $\beta$ that  minimizes RSS we take the derivative with respect to $\beta$, set it equal to zero and solve for $\beta$ as shown  next.

$$
\begin{equation}
\label{drdbeta}
\frac{\partial RSS}{\partial \beta} = -2 \pmb{X}^T(y - \pmb{X})
\end{equation}
$$
$$
\begin{equation}
\frac{\partial RSS}{\partial \beta \partial \beta^T} = 2\pmb{X}^T \pmb{X}
\end{equation}
$$

$$
\begin{equation}
\pmb{X^T} (\pmb{y}  - \pmb{X}\beta) = 0
\end{equation}
$$
The solution would be equation $\ref{beta_opt}$. For more details on this derivation, please refer to [1].  

$$
\begin{equation}
\color{green}
\label{beta_opt}
\hat{\beta} = (\pmb{X^T} \pmb{X})^{-1} \pmb{X}^T  \pmb{y}
\end{equation}
$$


We can use $\hat{\beta}$   back in equation  $\ref{lin_reg_eq}$. We result with equation $\ref{y_pred}$, where $\hat{y}$  is a $(N\times1)$ vector with  the predicted values at the training inputs. 

$$
\begin{equation}
\label{y_pred}
\hat{y} = \pmb{X} \hat{\beta} = \pmb{X}
(\pmb{X^T} \pmb{X})^{-1} \pmb{X}^T \pmb{y}
\end{equation}
$$

$$
\color{purple}
\begin{equation}
\underset{N\times 1}{\hat{y}} = \underset{N \times (p+1)}{\pmb{X}} \times \underset{(p + 1)\times 1}{\hat{\beta}}
\end{equation}
$$


- Explain how to use previous equation with one input
To determine the 

## Ridge
- ridge (3.41)

$$
\begin{equation}
\hat{\beta}^{ridge} = argmin_{\beta} \left\{ \sum_{i=1}^{N} \left(y_i - \beta_0 - \sum^{p}_{j = 1} X_j \beta_j \right)^2  + \lambda \sum_{j = 1}^{p} \beta_j^2  \right\}
\end{equation}
$$



- ridge (equivalent: 3.42)

$$
\begin{equation}
\hat{\beta}^{ridge} = argmin_{\beta} \sum_{i = 1}^{N} \left(y_i - \beta_0 - \sum_{j = 1}^{p} x_{ij}\beta_j\right)^2
\end{equation}
$$

Subject to 
$$
\begin{equation}
\sum_{j = 1}^{p} \beta_{j}^{2}  \leq t
\end{equation}
$$
- 3.43

$$
\begin{equation}
RSS(\lambda) = (\pmb{y} - \pmb{X}\beta)^T (\pmb{y} - \pmb{X}\beta) + \lambda \beta^T \beta
\end{equation}
$$

- 3.44

$$
\begin{equation}
\hat{\beta}^{ridge} =  (\pmb{X}^T \pmb{X} + \lambda\pmb{I}) ^{-1} \pmb{X}^T \pmb{y}
\end{equation}
$$


## The lasso
$$
\begin{equation}
\hat{\beta}^{lasso} = argmin_\beta  \sum_{i = 1}^{N} \left(y_i - \beta_0 - \sum_{j = 1}^{p} x_{ij} \beta_j \right)^2
\end{equation}
$$

Subject to 

$$
\begin{equation}
\sum_{j = 1}^{p} \vert \beta_{j} \vert \leq t
\end{equation}
$$

- 3.52

$$
\begin{equation}
\hat{\beta}^{lasso} = argmin_\beta  \left\{ \frac{1}{2} \sum_{i = 1}^{N} \left(y_i - \beta_0 - \sum_{j = 1}^{p}x_{ij} \beta_j\right)^2  + \lambda \sum_{j = 1}^{p} \vert \beta \vert \right\}
\end{equation}
$$

- Add and discuss fig 3.11


## Generalize ridge and lasso as Bayes estimates

$$
\begin{equation}
\tilde{\beta} = argmin_\beta  \left\{\sum_{i = 1}^{N} \left(y_i - \beta_0 - \sum_{j = 1}^{p}x_{ij} \beta_j\right)^2  + \lambda \sum_{j = 1}^{p} \vert \beta_j \vert^q \right\}
\end{equation}
$$


- Discuss 3.12 


## Elastic net 
$$
\begin{equation}
\lambda \sum_{j = 1}^{p} \left(\alpha \beta^{2}_j  + (1 - \alpha) \vert \beta_j \vert  \right)
\end{equation}
$$
- discuss  3.13



# Bibliography
[1] Hastie, Trevor, et al. _The Elements of Statistical Learning: Data Mining, Inference, and Prediction_. Springer, 2017.