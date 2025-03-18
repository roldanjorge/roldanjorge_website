---
author: "Jorge Roldan"
date: '2025-03-17'
title: '🔋 Pilas: Issue 2'
layout: "pilas"
categories: "newsletter"
draft: true
---

# Signal's President Meredith Whittaker warns about the security risks of AI Agents
Signal's president, Meredith Whittaker, raised concerns about the imminent dangers that AI agents pose to the privacy and security of devices. Meredith argued that granting AI agents access to multiple crucial services, apps,  and data in our devices could eventually threaten "to break the blood-brain barrier between the application and the Operating System (OS) layer," potentially undermining the privacy of secure applications, such as Signal [^agent_risks] .

# Google releases Gemma 3
Google released Gemma 3 on March 12th, 2025. This lightweight model runs on a single Graphical Processing Unit (GPU) or Tensor Processing Unit (GPU), which means it can potentially run locally on a device. It offers support for more than 140 languages. These are some highlights [^gemma_announcement] [^gemma_technical_report] [^gemma3_developer_guide] .


{{< figure id="gemma3_1" src="./gemma3_1.png" alt="gemma3_1" caption="Gemma 3 sizes and capabilities"  >}}

-  **Sizes**: Gemma 3 comes in 4 sizes: 1B, 4B, 12B, and 27B. Context length, languages, and input modalities are shown in {{< figref "gemma3_1" >}}.
-   **Architecture**: Gemma 3 uses a decoder-only transformer architecture leveraging Grouped-Query Attention [^gqa] with post-norm and pre-norm with RMSNorm [^rmsnorm] . Starting with a local layer, it uses a pattern of 5 local layers for every global layer. Gemma 3 uses a 400M vision encoder based on SigLIP encoder [^sigmoid_loss_language_image] . 
- **Pre-training**: The amount of tokens in Trillions (T) for the pre-training of versions 1B, 4B, 12B, and 27B was 2T, 4T, 12T, and 14T, respectively. Gemma 3 uses a SentencePiece tokenizer and leverages distillation. 
-  **Post-training**: Instruction tuning for Gemma 3 involved using knowledge distillation [^agarwal_distill] and Reinforcement Learning (RL) finetuning techniques based on BOND [^sessa_bond], WARM [^rame_warm], and WARP [^rame_warp] .


{{< figure id="gemma3_2" src="./gemma3_2.png" alt="gemma3_2" caption="Gemma 3 Benchmarking results"  >}}

- Gemma 3 ranks very well for its relatively small size compared to other models, achieving a 9th spot with a 1338 score in [Chatbot Arena](https://lmarena.ai/?leaderboard) [^chiang_chatbot_arena] as of March 8th, 2025. Other benchmark results are shown in {{< figref "gemma3_2" >}}. 


# OpenAI announces new tools for building AI agents 
OpenAI released on March 11th, 2025, a set of very powerful tools for building agents. The tools encompass the new [Responses API](https://platform.openai.com/docs/quickstart?api-mode=responses), built-in tools for search and computer use, a new agents SDK, and other observability tools [^openai_agent_tools] .

{{< figure id="openai_agent_tools" src="./openai_agent_tools.png" alt="openai_agent_tools" caption="OpenAI new tools to build AI Agents"  >}}

- **Response API**: This is a new API primitive that combines chat completions with the tool-use capabilities of the Assistants API and will support web search, file search, and computer use.
- **Built-in tools**: With web search, developers can ensure that model responses are relevant and up-to-date and retrieve information for documents using the file search tool. Finally, computer use empowers developers to use [Computer-Using Agent (CUA)](https://openai.com/index/computer-using-agent/) to create very powerful agents.
- **Agents SDK**: Allows developers to orchestrate workflows by allowing easy configurable LLMs with instructions and built-in tools. Includes mechanisms to transfer control between agents, provides guardrails, and tracing and observability tools. 


# Anthropic's new open protocol to connect LLMs with data sources and tools
- On Nov 25th, 2025, Anthropic announced a new open protocol that empowers developers to integrate LLMs with data sources and other tools. This protocol has gained popularity because of its power and simplicity [^mcp] [^mcp_intro] . The architecture consists of 5 components: MCP hosts, MCP clients, MCP servers, local data sources, and remote services. For a detailed over, visit [MCP's get started guide](https://modelcontextprotocol.io/introduction)

{{< figure id="mcp_architecture" src="./mcp_architecture.png" alt="mcp_architecture" caption="MCP Architecture "  >}}

# Andrej Karpathy shares how he uses LLMs 
It's always inspiring and energizing to watch Andrej Karpathy's lectures. This time, Andrej shares [how he leverages for day-to-day tasks](https://www.youtube.com/watch?v=EWvNQjAaOHw&t=147s) [^how_i_use_llms] . Some of the tools and services he covers are ChatGPT, Claude, Gemini, Cursor, NotebookLM, DALL-E, Sora, and many others.


# References
[^agent_risks]: SXSW, The State of Personal Online Security and Confidentiality | SXSW LIVE, (Mar. 07, 2025). Accessed: Mar. 12, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=AyH7zoP-JOg


[^video_understanding]: J. Jiang et al., “Token-Efficient Long Video Understanding for Multimodal LLMs,” Mar. 06, 2025, arXiv: arXiv:2503.04130. doi: 10.48550/arXiv.2503.04130.

[^manus]: “Manus.” Accessed: Mar. 11, 2025. [Online]. Available: https://manus.im/

[^gemma_technical_report]: Gemma Team and Google DeepMind1, “Gemma 3 Technical Report.” [Online]. Available: https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf


[^gemma3_developer_guide]: “Introducing Gemma 3: The Developer Guide- Google Developers Blog.” Accessed: Mar. 12, 2025. [Online]. Available: https://developers.googleblog.com/en/introducing-gemma3/

[^gemma_announcement]:“Introducing Gemma 3: The most capable model you can run on a single GPU or TPU,” Google. Accessed: Mar. 12, 2025. [Online]. Available: https://blog.google/technology/developers/gemma-3/


[^mcp]: “Introducing the Model Context Protocol.” Accessed: Mar. 12, 2025. [Online]. Available: https://www.anthropic.com/news/model-context-protocol


[^how_i_use_llms]: Andrej Karpathy, How I use LLMs, (Feb. 27, 2025). Accessed: Mar. 12, 2025. [Online Video]. Available: https://www.youtube.com/watch?v=EWvNQjAaOHw


[^machines_of_loving_grace]: D. Amodei, “Dario Amodei — Machines of Loving Grace.” Accessed: Mar. 12, 2025. [Online]. Available: https://darioamodei.com/machines-of-loving-grace


[^protocols_not_platforms]: “Protocols, Not Platforms: A Technological Approach to Free Speech,” Knight First Amendment Institute. Accessed: Mar. 12, 2025. [Online]. Available: http://knightcolumbia.org/content/protocols-not-platforms-a-technological-approach-to-free-speech


[^openai_agent_tools]: “New tools for building agents.” Accessed: Mar. 12, 2025. [Online]. Available: https://openai.com/index/new-tools-for-building-agents/


[^gqa]: J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai, “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints,” Dec. 23, 2023, arXiv: arXiv:2305.13245. doi: 10.48550/arXiv.2305.13245.

[^rmsnorm]: B. Zhang and R. Sennrich, “Root Mean Square Layer Normalization,” Oct. 16, 2019, arXiv: arXiv:1910.07467. doi: 10.48550/arXiv.1910.07467.

[^sigmoid_loss_language_image]: X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid Loss for Language Image Pre-Training,” Sep. 27, 2023, arXiv: arXiv:2303.15343. doi: 10.48550/arXiv.2303.15343.

[^agarwal_distill]: R. Agarwal et al., “On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes,” Jan. 17, 2024, arXiv: arXiv:2306.13649. doi: 10.48550/arXiv.2306.13649.


[^rame_warp]: A. Ramé et al., “WARP: On the Benefits of Weight Averaged Rewarded Policies,” Jun. 24, 2024, arXiv: arXiv:2406.16768. doi: 10.48550/arXiv.2406.16768.

[^rame_warm]: A. Ramé et al., “WARM: On the Benefits of Weight Averaged Reward Models,” Jan. 22, 2024, arXiv: arXiv:2401.12187. doi: 10.48550/arXiv.2401.12187.

[^sessa_bond]: P. G. Sessa et al., “BOND: Aligning LLMs with Best-of-N Distillation,” Jul. 19, 2024, arXiv: arXiv:2407.14622. doi: 10.48550/arXiv.2407.14622.


[^chiang_chatbot_arena]: W.-L. Chiang et al., “Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference,” Mar. 07, 2024, arXiv: arXiv:2403.04132. doi: 10.48550/arXiv.2403.04132.


[^mcp_intro]: “Introduction,” Model Context Protocol. Accessed: Mar. 17, 2025. [Online]. Available: https://modelcontextprotocol.io/introduction
