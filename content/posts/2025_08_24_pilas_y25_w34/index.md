---
author: "Jorge Roldan"
date: '2025-08-24'
title: '🔋Pilas: y25-w34'
categories: ['newsletter']
ShowToc: true
ShowBreadCrumbs: false
<<<<<<< HEAD
draft: true
---

# Models/Systems
=======
cover:
  image: "images/posts/2025_08_24_pilas_y25_w34/dinov3_benchmarks.png"
---

# Releases


## DINOv3
- Paper: [Hugging Face](https://huggingface.co/papers/2508.10104) [Arxiv](https://arxiv.org/abs/2508.10104)
- [Huggin Face Collection](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009)
- Blog post: [DINOv3: Self-supervised learning for vision at unprecedented scale](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)
- Website:  [Self-supervised learning for vision at unprecedented scale](https://ai.meta.com/dinov3/)

DINOv3 is a generalist, computer vision foundation model that scales self-supervised learning (SSL) and produces high-resolution visual features eliminating the need for labeled data.


{{< figure id="dinov3_benchmarks.png" src="./dinov3_benchmarks.png" alt="dinov3_benchmarks" caption="DINOv3 benchmarks" width="700"  >}}


## Intern-S1: A Scientific Multimodal Foundation Model
- Paper: [Hugging Face](https://huggingface.co/papers/2508.15763) -- [Arxiv](https://arxiv.org/abs/2508.15763)
- Models: [Intern-S1](https://huggingface.co/internlm/Intern-S1) -- [Intern-S1-mini](https://huggingface.co/internlm/Intern-S1-mini)

Intern-S1 is a large-scale multimodal Mixture-of-Experts (MoE) foundation model released by The Shanghai AI Laboratory. It is designed to close the gap between general-purpose open-source models and expert-level closed-source models in scientific domains. The model has 28 billion active parameters, 241 billion total parameters, and it was pretrained on 5T tokens. The authors used Mixture-of-Rewards (MoR), a novel RL technique to train simultaneously on more than 1000 tasks.

## Ovis2.5 
- Paper: [Hugging Face](https://huggingface.co/papers/2508.11737) -- [Arxiv: Ovis2.5 Technical Report](https://arxiv.org/abs/2508.11737)

Ovis2.5 is an open-source multimodal model released by Alibaba that introduces native-resolution vision and reflective reasoning. It achieves state-of-the-art performance in STEM, chart analysis, and multimodal benchmarks. 

# Thyme: Think Beyond Image
- Paper: [Hugging Face](https://huggingface.co/papers/2508.11630) -- [Arxiv](https://arxiv.org/abs/2508.11630)

Thyme enables multimodal LLMs to autonomously generate code for image manipulation and math, and with its GRPO-ATS training strategy, achieves strong gains on high-resolution perception and complex reasoning benchmarks.

# Organization Highlight
## Polymathic AI
- [Hugging Face Organization card](https://huggingface.co/polymathic-ai)
- [Polymathic-AI](https://polymathic-ai.org/)
- **Mission**: To usher in a new class of machine learning for scientific data, building models that can leverage shared concepts across disciplines. We aim to develop, train, and release such foundation models for use by researchers worldwide.
- [@PolymathicAI - X](https://x.com/PolymathicAI)
{{< figure id="polymathic.png" src="./polymathic.png" alt="Polymathic AI" caption="Polymathic-AI: Advancing Science through Multi‑Disciplinary AI" width="700"  >}}

- Datasets released:
  - [The Well](https://huggingface.co/collections/polymathic-ai/the-well-67e129f4ca23e0447395d74c)
    - A 15TB collection of physics simulation datasets.

# Notable Papers
- [Speed Always Wins: A Survey on Efficient Architectures for Large Language Models](https://arxiv.org/abs/2508.09834v1) - 08/13/25
- [MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training](https://arxiv.org/abs/2501.07556) - 01/13/2025
-  [QDataSet, quantum datasets for machine learning](https://www.nature.com/articles/s41597-022-01639-1) - 09/23/2022

# Repositories
- [gpt-oss](https://github.com/openai/gpt-oss)
    - gpt-oss-120b and gpt-oss-20b are two open-weight language models by OpenAI
- [ai-engineering-toolkit](https://github.com/Sumanth077/ai-engineering-toolkit)
    - A curated list of 100+ libraries and frameworks for AI engineers building with LLMs
- [Awesome-Efficient-Arch](https://github.com/weigao266/Awesome-Efficient-Arch) 
    - [Speed Always Wins: A Survey on Efficient Architectures for Large Language Models](https://arxiv.org/abs/2508.09834v1)

# Textbooks
- [Prompt Engineering for LLMs: The Art and Science of Building Large Language Model-Based Applications: Berryman, John, Ziegler](https://www.amazon.com/Prompt-Engineering-LLMs-Model-Based-Applications/dp/1098156153?&linkCode=sl1&tag=arcturuslabs-20&linkId=3e15a95d446ba84d7fb173e0e8a0ce15&language=en_US&ref_=as_li_ss_tl)


>>>>>>> main
