---
author: "Jorge Roldan"
date: '2025-07-07'
title: 'Building Agents with OpenAI'
categories: ['article']
ShowToc: true
ShowBreadCrumbs: false
draft: true
---


# OpenAI announces new tools for building AI agents 
OpenAI released on March 11th, 2025, a set of very powerful tools for building agents. The tools encompass the new [Responses API](https://platform.openai.com/docs/quickstart?api-mode=responses), built-in tools for search and computer use, a new agents SDK, and other observability tools [^openai_agent_tools] .


- **Response API**: This is a new API primitive that combines chat completions with the tool-use capabilities of the Assistants API and will support web search, file search, and computer use.
- **Built-in tools**: With web search, developers can ensure that model responses are relevant and up-to-date and retrieve information for documents using the file search tool. Finally, computer use empowers developers to use [Computer-Using Agent (CUA)](https://openai.com/index/computer-using-agent/) to create very powerful agents.
- **Agents SDK**: Allows developers to orchestrate workflows by allowing easy configurable LLMs with instructions and built-in tools. Includes mechanisms to transfer control between agents, provides guardrails, and tracing and observability tools. 
