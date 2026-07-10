# Sunset AI Agents

This template showcases a AI Agentic workflow implemented using [LangGraph](https://github.com/langchain-ai/langgraph), designed for [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio). This way the testing and debugging is totally transparent. It is based on the template repo https://github.com/langchain-ai/react-agent. It is personal project to showcase how I use LangGraph to implement AI Agents capable of using interesting tools to show the user a sunset depending of what is the available information and the time of the day when the interaction takes place.



## What it does

This workflow is currently in implementation but the desired scope is the following:

1. The 1st Tool enabled is Tavily for web search. The Human-In-The-Loop is meant to ask about sunsets in different parts of the world and the agent in charge should ask the user which location is the preferred one to continue. 
2. The 2nd Tool enabled is Windy MCP server running locally to provide a live stream from the selected location. WIP
3. This image result can then be analyzed by an agent that can describe in words with details what is characteristic of that place.
4. Then, a GEN AI step can generate a nice looking image with the current time composing something memorable as the result to the user.
5. Finally, another HITL step where the user can decide to add a rule for the creation of another image showing that the workflow uses memory to improve with each iteration and feedback.


## Getting Started

[Set up will be described at the end of implementation...]


## Development

[Overall description of the workflow to achieve the working software and references here...]

[^1]: https://python.langchain.com/docs/concepts/#tools

### Models

The intention of this project is also to change the LLMs used for the different steps described above. They should be different in terms of latency, quality and costs.
For this I will try:
1. ChatGPT mini models
2. Free OpenRouter models (TBD)
3. Free local models running on a separate PC (TBD)




