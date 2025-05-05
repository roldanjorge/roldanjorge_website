---
author: "Jorge Roldan"
date: '2025-04-27'
title: 'Pilas Issue 1'
# tags: ['ngram', 'language_models', 'nlp']
categories: ['newsletter']
ShowToc: true
ShowBreadCrumbs: false
draft: true
---


# 1️⃣ Model Releases
## OpenAI
### GPT-4.1, GPT-4.1 mini, GPT-4.1 nano - 04/14/25
[Announcement](https://openai.com/index/gpt-4-1/)
[^gpt4_1] 


### OpenAI o3 and o4-mini - 04/16/25
[Announcement](https://openai.com/index/introducing-o3-and-o4-mini/)
, [OpenAI o3 and o4-mini System Card](https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf)

OpenAI announced on April 16 the new reasoning models o3 and o4-mini. These are the main highlights: [^o3_o4_mini]

- o3 and o4-mini are the first models from OpenAI with access to all ChatGPT internal tools, they can use the browser, Python, and can do image and file analysis, and even customized tools using [function calling](https://platform.openai.com/docs/guides/function-calling?api-mode=responses). They use [chain-of-thought](https://www.ibm.com/think/topics/chain-of-thoughts) reasoning to decide when to use the tools to formulate their answers.
- These new models are taking chain-of-thought reasoning to the next level by thinking with images improving their capabilities of solving problems that require visual understanding, they included additional details [here](https://openai.com/index/thinking-with-images/)
- OpenAI continues to scale Reinforcement Learning leveraging chain-of-thought. OpenAI claims that o3 and o4-mini are also safer by being able to reason better about safety policies through deliberative alignment [^openai_deliberative_alignment]
- OpenAI shared multiple benchmark results for o3 and o4-mini in the [release announcement](https://openai.com/index/introducing-o3-and-o4-mini/) as well as safety and evaluations results in the [system card](https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf).


## Google
### Gemini 2.5 Pro - 03/25/25
[Announcement](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/), [Gemini 2.5 Pro Preview Model Card](https://storage.googleapis.com/model-cards/documents/gemini-2.5-pro-preview.pdf) 

Google announced on March 25 an experimental version of the Gemini 2.5 Pro thinking model. 
[^gemini_2_5]. Some of its key features include enhanced reasoning, advanced coding, multimodal input support for text, images, video, and audio, and a context window of 1-million tokens as well as text-only output with a maximum of 64K tokens. Gemini 2.5 Pro leverages the sparse Mixture-of-Experts (MoE) architecture, and it was trained using [JAX](https://github.com/jax-ml/jax), [ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)  and [Google's Tensor Processing Unites (TPUs)](https://cloud.google.com/tpu?e=48754805&hl=en).

{{< figure id="gemini_pro_benchmarks" src="./gemini_pro_benchmarks.png" alt="gemini_pro_benchmarks" caption="Gemini 2.5 Pro and benchmark results" >}}

{{< figref "gemini_pro_benchmarks" >}} shows Gemini 2.5 Pro's performance in common benchmarks.

### Gemini 2.5 Flash - 04/17/25
[Announcement](https://developers.googleblog.com/en/start-building-with-gemini-25-flash/) , [Gemini 2.5 Flash Preview Model Card](https://storage.googleapis.com/model-cards/documents/gemini-2.5-flash-preview.pdf)

Google launched their latest Gemini 2.5 Flash model through the Gemini API on April 17. This hybrid reason model allows its users to deliverately swith thinking on and off, and even to set thinking budgets. Similarly to Gemini 2.5 Pro, this model's architecture is based on a sparse Mixture of Experts (MoE) and it was trained using [JAX](https://github.com/jax-ml/jax), [ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)  and [Google's Tensor Processing Unites (TPUs)](https://cloud.google.com/tpu?e=48754805&hl=en). [^gemini_2_5_flash]


{{< figure id="gemini_2_5_flash_benchmarks" src="./gemini_2_5_flash_benchmarks.png" alt="gemini_2_5_flash_benchmarks" caption="Gemini 2.5 Flash cost and benchmark results" >}}

{{< figref "gemini_2_5_flash_benchmarks" >}} shows the Gemini 2.5 flash's performance in common benchmarks.

## Meta
### The Llama 4 Herd - 04/05/25
[Announcement](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), [Llama 4 - model card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/)

Meta released the Llama 4 herd on April 5 which consists of three open-weight multimodal models: Llama 4 Behemoth, Llama 4 Maverick and  Llama 4 Scout.  Maverick and Scout are available to download in [Huggingface](https://huggingface.co/meta-llama). However, Behemoth has not being released yet. Here are the main highlights: [^the_llama4_herd]
 [^llama4_model_card]

#### Model Sizes
The Llama 4 models leverage the mixture-of-experts (MoE) architecture which allows the model to only activate a subset of its total parameters during inference based on the selected expert. 

| Model            | Active Parameters | Total Parameters | Experts |
| ---------------- | ----------------- | ---------------- | ------- |
| Llama 4 Behemoth | 288B             | 2T               | 16      |
| Llama 4 Maverick | 17B               | 400B                | 128     |
| Llama 4 Scout    | 17B               | 109B                | 16     |

#### Multimodal by design
Both Maverick and scout accept text and a maximum of 5 images as inputs, and can only output text. The support the following languages Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnamese. However, image understanding is limited to English.

#### Inference Efficiency
Meta claims that the int4-quantized version of Llama 4 Scout fits in a single H100 GPU with a context length of 10M tokens while Llama 4 Maverick requires a "single H100 DGX host with distributed inference" and has a context length of 1M. One caveat about the context length capabilities is that they are "evaluated across 512 GPUs using 5D parallelism". 

#### Training techniques
- Llama 4 was pretrained on 200 languages. A total of more than 30 trillion tokens from a data mixture including text, image, and video datasets which is more than double the data used for Llama 3.
- Meta optimized training by using FP8 precision during pre-training with 32K GPUs, achieving 390 TFLOPs/GPU.
- Meta developed MetaP, a new training technique to set hyper-parameter values including per-layer learning rates and initialization scales. 
- Meta using this pipelines of post-training techniques lightweight supervised fine-tuning (SFT), online reinforcement learning (RL), and lightweight direct preference optimization (DPO).

## Adobe
### Firefly new Image Model 4 - 04/24/25
[^adobe_firefly]
 
## MidJourney
### V7 Alpha - 04/05/25
[^v7_alpha]

# 2️⃣ Agents

## Google's Agent2Agent Protocol (A2A)
[^google_agent2agent]


## Guides for Building AI Agents

### A practical guide to building agents by OpenAI 
- [Guide](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)

### Building effective agents by Anthropic
[^building_effective_ai_agents_anthropic]

# 3️⃣ Privacy/Security/Safety
## Cybersecurity Challenges of AI agents 
[^cyberattacks_by_ai_agents]

## Models generate exploits
[^ai_models_generate_exploits]

## DNA company 23&ME Bankrups 
[^23andme_brankrups], 
[^23andme_bankruptcy_cnbc]

## Waymo may use interior camera data to train generative AI models, but riders will be able to opt out
[^waymo_interior_camera]

## OpenAI slashes AI model safety testing time
[^openai_slashes_safety]

# 4️⃣ Awesome Textbooks
## AI Engineering book by Chip Huyen
[^ai_engineering_book]

## Reinforcement Learning from Human Feedback by Nathan Lambert  
[^rlhf_book_lambert]

# 5️⃣ Awesome Papers
## Welcome to the Era of Experience
[^welcome_to_the_era_of_experience]

## Advances and Challenges of Foundation Agents
[^advances_and_challenges_of_foundation_agents]

## Protocols not Platforms
[^protocols_not_platforms]

## A Comprehensive Review of Recommender Systems: Transitioning from Theory to Practice
[^rec_systems_review]

# References
[^gpt4_1]: “Introducing GPT-4.1 in the API.” Accessed: Apr. 27, 2025. [Online]. Available: https://openai.com/index/gpt-4-1/

[^o3_o4_mini]: “Introducing OpenAI o3 and o4-mini.” Accessed: Apr. 27, 2025. [Online]. Available: https://openai.com/index/introducing-o3-and-o4-mini/

[^gemini_2_5]: “Gemini 2.5: Our most intelligent AI model,” Google. Accessed: Apr. 27, 2025. [Online]. Available: https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/

[^gemini_2_5_flash]: “Start building with Gemini 2.5 Flash- Google Developers Blog.” Accessed: Apr. 27, 2025. [Online]. Available: https://developers.googleblog.com/en/start-building-with-gemini-25-flash/?utm_source=tldrai

[^cyberattacks_by_ai_agents]: “Cyberattacks by AI agents are coming,” MIT Technology Review. Accessed: Apr. 27, 2025. [Online]. Available: https://www.technologyreview.com/2025/04/04/1114228/cyberattacks-by-ai-agents-are-coming/

[^ai_models_generate_exploits]: T. Claburn, “AI models can generate exploit code at lightning speed.” Accessed: Apr. 27, 2025. [Online]. Available: https://www.theregister.com/2025/04/21/ai_models_can_generate_exploit/

[^v7_alpha]: “V7 Alpha,” Midjourney. Accessed: Apr. 27, 2025. [Online]. Available: https://www.midjourney.com/updates/website

[^advances_and_challenges_of_foundation_agents]: B. Liu et al., “Advances and Challenges in Foundation Agents: From Brain-Inspired Intelligence to Evolutionary, Collaborative, and Safe Systems,” Mar. 26, 2025, arXiv: arXiv:2504.01990. doi: 10.48550/arXiv.2504.01990.

[^the_llama4_herd]: “The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation,” Meta AI. Accessed: Apr. 27, 2025. [Online]. Available: https://ai.meta.com/blog/llama-4-multimodal-intelligence/

[^adobe_firefly]: A. F. Team, “Adobe Firefly: The next evolution of creative AI is here | Adobe Blog.” Accessed: Apr. 27, 2025. [Online]. Available: https://blog.adobe.com/en/publish/2025/04/24/adobe-firefly-next-evolution-creative-ai-is-here

[^ai_engineering_book]: “AI Engineering[Book].” Accessed: Apr. 27, 2025. [Online]. Available: https://www.oreilly.com/library/view/ai-engineering/9781098166298/


[^llama4_model_card]: “Llama 4 | Model Cards and Prompt formats.” Accessed: Apr. 27, 2025. [Online]. Available: https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/


[^building_effective_ai_agents_anthropic]: “Building Effective AI Agents.” Accessed: Apr. 27, 2025. [Online]. Available: https://www.anthropic.com/engineering/building-effective-agents


[^23andme_brankrups]: Breaking Points, 23&ME BANKRUPT: DNA SAMPLES FOR SALE, (Mar. 25, 2025). Accessed: Apr. 27, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=Y-2-TZzRJU0


[^rlhf_book_lambert]: N. Lambert, “Reinforcement Learning from Human Feedback”.

[^welcome_to_the_era_of_experience]: D. Silver and R. S. Sutton, “Welcome to the Era of Experience”.

[^waymo_interior_camera]: R. Bellan, “Waymo may use interior camera data to train generative AI models, but riders will be able to opt out,” TechCrunch. Accessed: Apr. 27, 2025. [Online]. Available: https://techcrunch.com/2025/04/08/waymo-may-use-interior-camera-data-to-train-generative-ai-models-sell-ads/


[^openai_slashes_safety]: C. Criddle, “OpenAI slashes AI model safety testing time,” Financial Times, Apr. 11, 2025.

[^23andme_bankruptcy_cnbc]: K. Williams, “23andMe bankruptcy: With America’s DNA put on sale, market panic gets a new twist,” CNBC. Accessed: Apr. 27, 2025. [Online]. Available: https://www.cnbc.com/2025/03/30/23andme-bankruptcy-selling-deleting-dna-genetic-testing.html


[^google_agent2agent]: “Announcing the Agent2Agent Protocol (A2A)- Google Developers Blog.” Accessed: Apr. 27, 2025. [Online]. Available: https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/?utm_source=tldrai

[^protocols_not_platforms]: “Protocols, Not Platforms: A Technological Approach to Free Speech,” Knight First Amendment Institute. Accessed: Apr. 27, 2025. [Online]. Available: http://knightcolumbia.org/content/protocols-not-platforms-a-technological-approach-to-free-speech

[^rec_systems_review]: S. Raza et al., “A Comprehensive Review of Recommender Systems: Transitioning from Theory to Practice,” Feb. 23, 2025, arXiv: arXiv:2407.13699. doi: 10.48550/arXiv.2407.13699.

[^openai_deliberative_alignment]: M. Y. Guan et al., “Deliberative Alignment: Reasoning Enables Safer Language Models,” Jan. 08, 2025, arXiv: arXiv:2412.16339. doi: 10.48550/arXiv.2412.16339.
