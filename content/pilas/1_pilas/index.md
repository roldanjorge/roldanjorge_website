---
author: "Jorge Roldan"
date: '2025-03-08'
title: 'Pilas Issue 1'
layout: "pilas"
---

#  1. Pioneers of Reinforcement Learning recieve the ACM A.M. Turing Award 2024
The ACM A.M. Turing Award 2024, commonly referred to as the "Nobel Prize in Computing",  was announced on March 5th, 2025 and it was awarded to [Andrew Barto](https://people.cs.umass.edu/~barto/), and [Richard Sutton](http://incompleteideas.net/) recognizing them as the pioneers of RL (Reinforcement Learning). Reinforcement Learning is one of the core pilars of AI and it studies how agents interacts with an environment to learn how to make better decisions. RL, and deep reinforcement learning have been at the core of many advancements including AlphaGo, ChatGPT, and other state-of-the-art LLMs. [^acm_turing_award]


{{< figure id="model_files" src="./acm_2024_award.png" alt="acm_2024_award" caption="X announcement of The ACM A.M. Turing Award 2024" width="" height="500" >}}


# 2. OpenAI launches GPT-4.5

OpenAI pushes the frontier of unsupervised learning with the new GPT-4.5 model which builds on GPT-4o. This model scales the learning paradigm as opposed to the reasoning paradigm like OpenAI o1, or o3-mini [^gpt4_5] . OpenAI published additional details about this model in the [OpenAI GPT-4.5 System Card](https://cdn.openai.com/gpt-4-5-system-card-2272025.pdf?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email). Regarding training, OpenAI "developed new, scalable alignment techniques", combined with supervised fine-tuning (SFT), as well as Reinforcement Learning from Human Feedback (RLHF). Unfortunately, not further details are included on model architecture or size. The system card presents multiple benchmark results showing promising overall improvements but also some risk increases. Finally, GPT 4.5 shows promising results in accuracy and reduction in hallucinations being the largest, and most knowledgeable model so far.

{{< figure id="model_files" src="./gpt_4_5.png" alt="gpt_4_5" caption="Accuracy and Hallucination rate of GPT-4.5" width="500" height="250" >}}



# 3. Anthropic's Claude 3.7  Sonnet

Anthropic announced Claude 3.7 Sonnet, their best model to date with hybrid reasonining capabilities. Anthropic also introduced Claude Code, a command line tool for agentic coding. Some details include [^claude_3_7] :
- Combined LLM capabilities and reasoning functionality allowing the user when to take longer while reasoning. Claude 3.7 Sonnet leverages  "extended thinking" mode by generating tokens to reason about a problem in depth before generating the final answer. 
- Trained on a mix of public and proprietary data with a knowledge cut-off date of October 2024
- Trained to be helpful, harmless, and honest. Training techniques included word prediction on large dataset as well as human feedback. The technique Constitutional AI was used to align the model with human values.

{{< figure id="model_files" src="./claude_3_7.png" alt="claude_3_7" caption="Performance of Claude 3.7" >}}

# 4. Mistral OCR

Mistral announced on March 6th, 2025 their state-of-the-art Optical Character Recognition (OCR) model offered through the *mistral-ocr-latest*  API at a cost of 1000 pages / $ [^mistral_ocr] . 
These are some of the highlights:
 
{{< figure id="mistral_ocr_performance" src="./mistral_ocr_performance.png" alt="mistral_ocr_performance" caption="Performance of Mistral OCR" >}}

- Mistral OCR's astonishing overall performance is 94.89, followed by Gemini-1.5-Flash-002 at 90.23 as shown in {{< figref "mistral_ocr_performance" >}}. I am particular impressed by the 94.29 performance in math considering how hard it is to correctly recognize Latex expressions.

- Given its impressive performance, Mistral OCR's shows promising potential for Retrieval Augment Generation (RAG) uses cases that leverage multimodal documents as inputs.

- Support for multiple languages includes ru, fr, hi, zh, pt, de, es, tr, uk, it, ro with an performance of at least 90. 

- Faster performancing processing up to 2000 pages per minute.



# 5. $DEMO^3$
- [Multi-Stage Manipulation with Demonstration-Augmented Reward, Policy, and World Model Learning](https://adrialopezescoriza.github.io/demo3/?utm_source=tldrai)


# 7. Llama Stack
- [llama-stack/docs/zero_to_hero_guide at main · meta-llama/llama-stack](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide?utm_source=tldrai)


# 8. Aya Vision
- [Aya Vision: Expanding the worlds AI can see](https://cohere.com/blog/aya-vision?utm_source=tldrai)

# 9. UniTok
- [2502.20321 - UniTok: A Unified Tokenizer for Visual Generation and Understanding](https://arxiv.org/abs/2502.20321?utm_source=tldrai)


#  10. Advancing healthcare and scientific discovery with AI
- [How Google Research is making healthcare more accessible and personalized with AI](https://blog.google/technology/health/google-research-healthcare-ai/?utm_source=tldrai)

# References
[^acm_turing_award]: “ACM A.M. Turing Award Honors Two Researchers Who Led the Development of Cornerstone AI Technology.” Accessed: Mar. 09, 2025. [Online]. Available: https://www.acm.org/media-center/2025/march/turing-award-2024?utm_source=tldrai

[^gpt4_5]: “Introducing GPT-4.5.” Accessed: Mar. 09, 2025. [Online]. Available: https://openai.com/index/introducing-gpt-4-5/


[^claude_3_7]: “Claude 3.7 Sonnet and Claude Code.” Accessed: Mar. 09, 2025. [Online]. Available: https://www.anthropic.com/news/claude-3-7-sonnet


[^mistral_ocr]: “Mistral OCR | Mistral AI.” Accessed: Mar. 09, 2025. [Online]. Available: https://mistral.ai/en/news/mistral-ocr
