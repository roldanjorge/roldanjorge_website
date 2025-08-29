---
author: "Jorge Roldan"
date: '2025-08-29'
title: 'The math of language models'
categories: ['article']
ShowToc: true
ShowBreadCrumbs: false
cover:
  image: "images/posts/2025_08_24_pilas_y25_w34/dinov3_benchmarks.png"
---

# Introduction

## Review

- [Attention is All you Need Paper](https://arxiv.org/abs/1706.03762)
    - **The Transformer Architecture**
        
        ![image.png](the_math_of_lms%2023264133769d80299b16cebf04996bc9/image.png)
        
    - Self Attention and Multi-Head Attention
        
        ![Image source: Attention is All you need](the_math_of_lms%2023264133769d80299b16cebf04996bc9/image%201.png)
        
        Image source: Attention is All you need
        
        ## Self-Attention Head
        
        ### Attention is All You Need
        
        $$
        \begin{equation} 
        Attention(Q, K, V) = softmax \Bigg( \frac{Q K^T}{\sqrt{d_k}} \Bigg) V
        \end{equation}
        $$
        

## Goal

<aside>
1️⃣

**Better understand the math behind a single attention head  with a simple example**

Why focus on math?

1. Math is beautiful.
2. Math is empowering.
</aside>

# **The math of a Self-Attention Head**

<aside>
📔

Note: This work was based on Chapter 9 - The Transformer - https://web.stanford.edu/~jurafsky/slp3/

</aside>

<aside>
📔

Note: All animations were created  for this lecture using the open source [manim](https://www.manim.community/) library. Thanks to the awesome manim-community and Grant Sanderson (**3Blue1Brown)**

</aside>

## Representing an input token

- Embedding representation
    
    ![image.png](the_math_of_lms%2023264133769d80299b16cebf04996bc9/image%202.png)
    
- Vector definition
    
    A vector  $\mathbf{x} \in \mathbb{R}^{1xd}$   is an ordered collection of  $d$ real numbers, written as:
    
    $$
    \begin{equation}
    \underbrace{\begin{bmatrix}x_{1,1} & x_{1,2} & \dots & x_{1,d}\\\end{bmatrix}}_{\mathbf{x}\;(1 \times d)}
    \end{equation}
    $$
    
    where each $x_{i,j} \in \mathbb{R}$
    

## A simple example: computing self-attention $a^{(3)}$  for $x^{(3)}$

- Computing self-attention example
    
    ![image.png](the_math_of_lms%2023264133769d80299b16cebf04996bc9/image%203.png)
    

## Roadmap

- Roadmap
    
    **1. Generate Key Query and Value Vectors**
    
    $$
    \begin{equation}\mathbf{q}_i = \mathbf{x}_i \mathbf{W}^Q\qquad\mathbf{k}_j = \mathbf{x}_j \mathbf{W}^K\qquad\mathbf{v}_j = \mathbf{x}_j \mathbf{W}^V\end{equation}
    $$
    
    **2. Compare  $x^{(3)}$ ‘s query with the keys for  $x^{(1)}$ , $x^{(2)}$ and $x^{(3)}$** 
    
    $$
    \begin{equation} 
    \textbf{q}_i \cdot \textbf{k}_j
    \end{equation}
    $$
    
    3. Divide scalar score by  $\sqrt{d_k}$
    
    $$
    \begin{equation}  score(\textbf{x}_i, \textbf{x}_j) = \frac{\textbf{q}_i \cdot \textbf{k}_j}{\sqrt{d_k}}\end{equation}
    $$
    
    **4. Turn into $\alpha_{i, j}$ weights via Softmax**
    
    $$
    \begin{equation}  \alpha_{ij} = softmax(score(\textbf{x}_i, \textbf{x}_j))  \ \forall j \leq i\end{equation}
    $$
    
    **5. Weight each value vector**
    
    $$
    \alpha_{ij} \textbf{v}_j \ \ \ \forall j \leq i
    $$
    
    **6. Sum the weighted value vectors**
    
    $$
    \begin{equation}  \textbf{head}_i = \sum_{j \leq i} \alpha_{ij} \textbf{v}_j\end{equation}
    $$
    
    **7. Reshape and output attention**
    
    $$
    
    \begin{equation}  \textbf{a}_i = \textbf{head}_i \textbf{W}^O\end{equation}
    $$
    

## 1. Generate Key Query and Value Vectors

### Matrix theory

- 📖 Matrix Definition
    
    <aside>
    📖
    
    A matrix  $\mathbf{W} \in \mathbb{R}^{d \times d_k}$ is a rectangular array of real numbers with $d$ rows and $d_k$ columns:
    
    $$
    \begin{equation}
    \mathbf{W} =\begin{bmatrix}w_{1,1} & w_{1,2} & \dots  & w_{1,n} \\w_{2,1} & w_{2,2} & \dots  & w_{2,n} \\\vdots & \vdots & \ddots & \vdots \\w_{d,1} & w_{d,2} & \dots  & w_{d,d_k}\end{bmatrix}
    \end{equation}
    $$
    
    where each $w_{ij} \in \mathbb{R}$
    
    </aside>
    
- 📖 Matrix Multiplication
    
    <aside>
    📖
    
    $$
    \begin{equation}
    \mathbf{q} = \mathbf{x}\mathbf{W}
    
    \end{equation}
    $$
    
    where $\mathbf{x}\in\mathbb{R}^{1 \times d},\;
    \mathbf{W}\in\mathbb{R}^{d \times d_k},\;
    \mathbf{q}\in\mathbb{R}^{1 \times d_k}$
    
    $$
    \underbrace{\begin{bmatrix}x_{1,1} & x_{1,2} & x_{1,3}\\\end{bmatrix}}_{\mathbf{x}\;(1\times3)}\;\underbrace{\begin{bmatrix}w_{1,1} & w_{1,2}\\w_{2,1} & w_{2,2}\\w_{3,1} & w_{3,2}\end{bmatrix}}_{\mathbf{B}\;(3\times2)}=\underbrace{\begin{bmatrix}x_{1,1}w_{1,1}+x_{12}w_{21}+x_{13}w_{31} &x_{1,1}w_{1,2}+x_{12}w_{22}+x_{13}w_{32}\\[4pt]\end{bmatrix}}_{\mathbf{C}\;(1\times2)}
    $$
    
    </aside>
    
- 🎬 Matrix Multiplication Animation
{{< youtube jL9VG4XSRlU >}}

### The Meaning of query, key and value vectors

**Query** ($\textbf{q}$):  Captures what this position is looking for in other positions.

**Key** ($\textbf{k}$): Represents what each position offers.

**Value** ($\textbf{v}$):  Contains the actual information to aggregate.

### Meaning In the context of our example

- In the context of our example when calculating attention for $a^{(3)}$
    1. Token 3 asks a question (its **query**).
    2. Every token responds with “Here’s the kind of question I can answer” (its **key**).
    3. If token 3 likes the answer (high similarity q·k), it listens to that token’s **value**—the detailed info it contributes.

### Generating key, query, and value vectors

$$
\begin{equation}\mathbf{q}^{(i)} = \mathbf{x}^{(i)} \mathbf{W}^Q\qquad\mathbf{k}^{(j)} = \mathbf{x}^{(j)} \mathbf{W}^K\qquad\mathbf{v}^{(j)} = \mathbf{x}^{(j)} \mathbf{W}^V\end{equation}
$$

- Where
    - Dimension of  $\textbf{x}^{(i)}: [1, d]$
    - Dimensions of $\textbf{q}^{(i)}, \textbf{k}^{(j)} \ : [1, d_k]$
    - Dimensions of  $\textbf{v}^{(j)} \ : [1, d_v]$
    - Dimensions of  $\mathbf{W}^Q, \mathbf{W}^K$: $[d, d_k]$
    - Dimensions of  $\mathbf{W}^V$: $[d, d_v]$
- For the original transformer paper:
    - $d = 512$
    - $d_k = d_v = 64$
- animation
    <!-- [https://www.youtube.com/watch?v=SLWrevk6_Us](https://www.youtube.com/watch?v=SLWrevk6_Us) -->
    
{{< youtube SLWrevk6_Us >}}

## 2. Compare  $x^{(3)}$ ‘s query with the keys for  $x^{(1)}$ , $x^{(2)}$ and $x^{(3)}$

- 📖 Dot product definition
    
    <aside>
    📖
    
     Given two vectors $\bm{q}, \bm{k} \in \mathbb{R}^{1,d_k}$, their dot product is defined as:
    
    $$
    \bm{q} \cdot \bm{k} = \sum_{i=1}^n q_i v_i
    $$
    
    $$
    \begin{align*}\bm{q}\cdot\bm{k}  &= \underbrace{q_1v_1 + q_2v_2 + q_3v_3}_{\text{sum form}} \\[6pt]  &= \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \!\cdot\!     \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}     \;=\; 1\!\cdot\!4 + 2\!\cdot\!5 + 3\!\cdot\!6     \;=\; 32                                   \quad     \bigl\lbrace\text{numeric example}\bigr\rbrace\end{align*}
    $$
    
    </aside>
    

$$
\begin{equation} 
\textbf{q}_i \cdot \textbf{k}_j
\end{equation}
$$

- Where
    - Dimension of $\textbf{q}_i \cdot \textbf{k}_j$: scalar
    - $j$ can be (1, 2, 3)
    - $i$ is 3

Note: We will have a scalar value for each comparison (a total of 3 values)

- animation
    
    <!-- [https://www.youtube.com/watch?v=Cx_jClD7L-A](https://www.youtube.com/watch?v=Cx_jClD7L-A) -->
 
{{< youtube Cx_jClD7L-A >}}

## 3. Divide scalar score by  $\sqrt{d_k}$

$$
\begin{equation}  score(\textbf{x}_i, \textbf{x}_j) = \frac{\textbf{q}_i \cdot \textbf{k}_j}{\sqrt{d_k}}\end{equation}
$$

- Where
    - Dimension of $\frac{\textbf{q}_i \cdot \textbf{k}_j}{\sqrt{d_k}}$: scalar
    - $j$ can be (1, 2, 3)
    - $i$ is 3

Note: We will have a scalar value for each comparison (a total of 3 values)

- animation
    
    <!-- [https://www.youtube.com/watch?v=QLXT6NvHPII](https://www.youtube.com/watch?v=QLXT6NvHPII) -->
    
{{< youtube QLXT6NvHPII >}}

## 4. Turn into $\alpha_{i, j}$ weights via Softmax

- 📖 $Softmax$ definition
    
    <aside>
    📖
    
    Given a vector $\mathbf{z} = [z_1, z_2, \dots, z_K] \in \mathbb{R}^K$, the softmax $\sigma$ function is defined as:
    
    $$
    \sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad \text{for } i = 1, \dots, K
    $$
    
    - $\sigma(\mathbf{z})_i > 0$ for all $i$
    - $\sum_{i=1}^K \sigma(\mathbf{z})_i = 1$
    </aside>
    

$$
\begin{equation}  \alpha_{ij} = softmax(score(\textbf{x}_i, \textbf{x}_j))  \ \forall j \leq i\end{equation}
$$

- Where:
    - Dimension of $\alpha_{ij}$: scalar
    - $j$  can be (1, 2, 3)
    - $i$ is 3
- animation
    
    <!-- [https://www.youtube.com/watch?v=V0SkmVAXWRI](https://www.youtube.com/watch?v=V0SkmVAXWRI) -->
    
{{< youtube V0SkmVAXWRI>}}

Note: We will have a scalar value for each comparison (a total of 3 values)

## 5. Weight each value vector

$$
\alpha_{ij} \textbf{v}_j \ \ \ \forall j \leq i
$$

- Where:
    - Dimension of $\alpha_{ij} \textbf{v}_j$:  $[1, d_v]$
    - $j$  can be (1, 2, 3)
    - $i$ is 3

Note: We will have a total of three vectors

- animation
    
    <!-- [https://www.youtube.com/watch?v=vCc9SPIBWSY](https://www.youtube.com/watch?v=vCc9SPIBWSY) -->
    
{{< youtube vCc9SPIBWSY>}}

## 6. Sum the weighted value vectors

$$
\begin{equation}  \textbf{head}_i = \sum_{j \leq i} \alpha_{ij} \textbf{v}_j\end{equation}
$$

- Where:
    - Dimension of $\textbf{head}_i$:  $[1, d_v]$
    - $j$  can be (1, 2, 3)
    - $i$ is 3

Note: We will have a total of 1 vector

- animation
    
    <!-- [https://www.youtube.com/watch?v=2mlKY-0i6b8](https://www.youtube.com/watch?v=2mlKY-0i6b8) -->
    
{{< youtube 2mlKY-0i6b8 >}}


## 7. Reshape and output attention

$$

\begin{equation}  \textbf{a}^{(i)} = \textbf{head}^{(i)} \textbf{W}^O\end{equation}
$$

- Where:
    - Dimension of $\textbf{head}_i$:  $[1, d_v]$
    - Dimension of $\textbf{W}^O$: $[d_v, d]$
    - Dimension of $\textbf{a}^{(i)}$ : $[1, d]$

Note: We will have a total of 1 vector

- animation
    
    <!-- [https://www.youtube.com/watch?v=0fx4-wibHJA](https://www.youtube.com/watch?v=0fx4-wibHJA) -->

{{< youtube 0fx4-wibHJA >}}