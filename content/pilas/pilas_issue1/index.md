---
author: "Jorge Roldan"
date: '2025-03-08'
title: 'Pilas Issue 1'
layout: "pilas"
---

#  Pioneers of Reinforcement Learning receive the 2024 ACM A.M. Turing Award
The ACM A.M. Turing Award 2024, commonly referred to as the "Nobel Prize in Computing," was announced on March 5th, 2025, and it was awarded to [Andrew Barto](https://people.cs.umass.edu/~barto/), and [Richard Sutton](http://incompleteideas.net/) recognizing them as the pioneers of Reinforcement Learning (RL). Reinforcement Learning is one of the core pillars of AI, and it studies how agents interact with an environment to learn how to make better decisions. RL and deep reinforcement learning have been at the core of many advancements, including AlphaGo, ChatGPT, and other state-of-the-art LLMs. [^acm_turing_award]


{{< figure id="model_files" src="./acm_2024_award.png" alt="acm_2024_award" caption="X announcement of The ACM A.M. Turing Award 2024" width="" height="500" >}}


# OpenAI launches GPT-4.5

OpenAI pushes the frontier of unsupervised learning with the new GPT-4.5 model, which builds on GPT-4o. This model scales the learning paradigm instead of the reasoning paradigm like OpenAI o1 or o3-mini [^gpt4_5] . OpenAI published additional details about this model in the [OpenAI GPT-4.5 System Card](https://cdn.openai.com/gpt-4-5-system-card-2272025.pdf?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email). Regarding training, OpenAI "developed new, scalable alignment techniques", combined with supervised fine-tuning (SFT), as well as Reinforcement Learning from Human Feedback (RLHF). Unfortunately, no further details are included on model architecture or size. The system card presents multiple benchmark results showing promising overall improvements but also some risk increases. Finally, GPT 4.5 shows promising results in accuracy and reduction in hallucinations, being the largest and most knowledgeable model so far.


{{< figure id="model_files" src="./gpt_4_5.png" alt="gpt_4_5" caption="Accuracy and Hallucination rate of GPT-4.5" width="500" height="250" >}}



# Anthropic's Claude 3.7  Sonnet

Anthropic announced Claude 3.7 Sonnet, their best model with hybrid reasoning capabilities. Anthropic also introduced Claude Code, a command line tool for agentic coding. Some details include [^claude_3_7] :
- Combined LLM capabilities and reasoning functionality, allowing the user when to take longer while reasoning. Claude 3.7 Sonnet leverages the "extended thinking" mode by generating tokens to reason about a problem in depth before generating the final answer. 
- Trained on a mix of public and proprietary data with a knowledge cut-off date of October 2024
- Trained to be helpful, harmless, and honest. Training techniques included word prediction on large datasets and human feedback. Constitutional AI was used to align the model with human values.

{{< figure id="model_files" src="./claude_3_7.png" alt="claude_3_7" caption="Performance of Claude 3.7" >}}

# Mistral OCR

Mistral announced on March 6th, 2025, their state-of-the-art Optical Character Recognition (OCR) model offered through the *mistral-ocr-latest*  API for 1000 pages / $ [^mistral_ocr] . 
These are some of the highlights:
 
{{< figure id="mistral_ocr_performance" src="./mistral_ocr_performance.png" alt="mistral_ocr_performance" caption="Performance of Mistral OCR" >}}

- Mistral OCR's astonishing overall performance is 94.89, followed by Gemini-1.5-Flash-002 at 90.23 as shown in {{< figref "mistral_ocr_performance" >}}. I am particularly impressed by the 94.29 performance in math, considering how hard it is to recognize Latex expressions correctly.

- Given its impressive performance, Mistral OCR shows promising potential for Retrieval Augment Generation (RAG) use cases that leverage multimodal documents as inputs.

- Support for multiple languages includes ru, fr, hi, zh, pt, de, es, tr, uk, it, ro with an performance of at least 90. 

- Faster performance processing up to 2000 pages per minute.


# Google's push to advance healthcare with AI

Yossi Matias, Google's VP & head of Google Research, discussed during the [Lake Nona Impact Forum](https://lakenonaimpactforum.org/event/overview/) the AI breakthroughs by the company in healthcare as it relates to accessibility, personalization, and effectiveness: [^google_healthcare_ai]

-  Google is improving access to high-quality medical content for all its users by building AI tools to help YouTube's health creators with their video creation process [^youtube_ai_tools_healthcare] . With [Google Lens](https://blog.google/products/google-lens/google-lens-features/), users can check their skin to identify skin-related conditions  [^google_lens], and tools like [MedLM](https://cloud.google.com/blog/topics/healthcare-life-sciences/introducing-medlm-for-the-healthcare-industry) and [Search for Healthcare](https://www.googlecloudpresscorner.com/2024-10-17-Google-Cloud-Launches-General-Availability-of-Vertex-AI-Search-for-Healthcare-and-Healthcare-Data-Engine) can directly answer user's questions.
Google is also betting on leveraging Generative AI for personalized healthcare with Med-Gemini [^med_gemini], a fine-tuned version of the Gemini models for medical applications and multimodal support. Promising performance on 14 medical benchmarks illustrates the power of these models to augment medical doctors' effectiveness in giving more personalized medicine around the globe [^med_gemini_capabilities] .
- Google's work on improving healthcare expand multiple streams such as developing technologies for better detecting [breast cancer](https://blog.google/technology/ai/icad-partnership-breast-cancer-screening/), [lung cancer](https://research.google/blog/computer-aided-diagnosis-for-lung-cancer-screening/) and [diabetic retinopathy](https://blog.google/around-the-globe/google-asia/arda-diabetic-retinopathy-india-thailand/) and other initiatives such as the [Health AI Developer Foundations](https://research.google/blog/helping-everyone-build-ai-for-healthcare-applications-with-open-foundation-models/), and the [Open Health Stack](https://blog.google/technology/health/open-health-stack-developers/).

{{< figure id="med_gemini" src="./med_gemini.png" alt="med_gemini" caption="Med-Gemini Development and benchmarking (K. Saab et al., “Capabilities of Gemini Models in Medicine”)" >}}

# Aya Vision
Cohere announced on March 3rd, 2025, the release of a state-of-the-art open-weights vision model focusing on bridging the gap between the performance of models in different languages. Aya Vision outperforms other models such as Gemini-Flash, Llama-3.2, and Pistral with smaller and computing efficient alternatives [^aya_vision] . You can access both the [8b](https://huggingface.co/CohereForAI/aya-vision-8b?ref=cohere-ai.ghost.io) and [32b](https://huggingface.co/CohereForAI/aya-vision-32b?ref=cohere-ai.ghost.io) the open-weights models in [Huggingface](https://huggingface.co/collections/CohereForAI/c4ai-aya-vision-67c4ccd395ca064308ee1484). 




{{< figure id="aya_vision" src="./aya_vision.png" alt="aya_vision" caption="Aya Vision 8B Win Rates" >}}


# References
[^acm_turing_award]: “ACM A.M. Turing Award Honors Two Researchers Who Led the Development of Cornerstone AI Technology.” Accessed: Mar. 09, 2025. [Online]. Available: https://www.acm.org/media-center/2025/march/turing-award-2024?utm_source=tldrai

[^gpt4_5]: “Introducing GPT-4.5.” Accessed: Mar. 09, 2025. [Online]. Available: https://openai.com/index/introducing-gpt-4-5/


[^claude_3_7]: “Claude 3.7 Sonnet and Claude Code.” Accessed: Mar. 09, 2025. [Online]. Available: https://www.anthropic.com/news/claude-3-7-sonnet


[^mistral_ocr]: “Mistral OCR | Mistral AI.” Accessed: Mar. 09, 2025. [Online]. Available: https://mistral.ai/en/news/mistral-ocr


[^google_healthcare_ai]: “Advancing healthcare and scientific discovery with AI,” Google. Accessed: Mar. 10, 2025. [Online]. Available: https://blog.google/technology/health/google-research-healthcare-ai/


[^youtube_ai_tools_healthcare]: “Exploring how AI tools can help increase high-quality health content,” blog.youtube. Accessed: Mar. 10, 2025. [Online]. Available: https://blog.youtube/inside-youtube/ai-tools-high-quality-health-content/


[^google_lens]: “8 ways Google Lens can help make your life easier,” Google. Accessed: Mar. 10, 2025. [Online]. Available: https://blog.google/products/google-lens/google-lens-features/


[^med_gemini]: “Advancing medical AI with Med-Gemini.” Accessed: Mar. 10, 2025. [Online]. Available: https://research.google/blog/advancing-medical-ai-with-med-gemini/


[^med_gemini_capabilities]: K. Saab et al., “Capabilities of Gemini Models in Medicine,” May 01, 2024, arXiv: arXiv:2404.18416. doi: 10.48550/arXiv.2404.18416.


[^must3r]: Y. Cabon et al., “MUSt3R: Multi-view Network for Stereo 3D Reconstruction,” Mar. 03, 2025, arXiv: arXiv:2503.01661. doi: 10.48550/arXiv.2503.01661.

[^aya_vision]: “Aya Vision: Expanding the worlds AI can see,” Cohere. Accessed: Mar. 10, 2025. [Online]. Available: https://cohere.com/blog/aya-vision
