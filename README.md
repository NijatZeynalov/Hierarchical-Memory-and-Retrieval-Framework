# Hierarchical Memory and Retrieval Framework

## Overview

This project is about creating an intelligent virtual agent that can **explore environments, remember important information, and make decisions based on those memories**. The agent has been designed to combine **navigation, memory management, and natural language understanding**, making it more effective in understanding and interacting with its surroundings.


The project combines three main components to make this happen:

- **Hierarchical Memory System**: A way for the agent to remember things in a structured way. The memory is organized hierarchically, which helps the agent recall details effectively and navigate efficiently. It stores various kinds of data, from locations to semantic details about what objects are in each room.

- **Natural Language Understanding**: The agent uses a language encoder to make sense of human-like instructions. For example, it can understand commands like "Go to the room with the blue chair" and use its memory to figure out where to go.

- **Navigation System**: The agent uses a **path planning algorithm** to navigate through the environment. It considers the layout of the environment, avoids obstacles, and follows instructions using both memory and sensor input.


## Project Structure

- **Configs**: Configuration files (`agent_config.yaml` and `memory_config.yaml`) provide settings for agent parameters, memory management, navigation preferences, and logging.
- **Memory Module**: Contains files for managing memory nodes and the hierarchical memory structure, including components like `SemanticForest` and `MemoryNode` that represent the core of memory management.
- **Navigation Module**: Responsible for generating actions and planning paths to reach goals within the environment.
- **Language Module**: Handles encoding of user commands into embeddings that can be used by the memory and retrieval systems to understand what the agent needs to do.
- **Retrieval Module**: Helps the agent find relevant information in its memory based on user queries or the agent's current tasks.

## How It Works

1. **Exploring the Environment**: The agent uses sensors (like cameras) to gather information. This information is stored in its memory using a structure called the `SemanticForest`.
2. **Remembering Details**: When the agent encounters something new, it stores it as a memory node, including spatial and semantic information. The agent can also prune old or irrelevant memories to stay efficient.
3. **Responding to Queries**: When given a command like "Go to the kitchen," the agent uses its **Text Encoder** to understand the request and then queries its memory to find the best path to fulfill the command.
4. **Navigation**: The agent plans a path to the target using a path planning algorithm. It combines both the memory of what it knows about the environment and the real-time sensor input to avoid obstacles and navigate effectively.

## Getting Started

### Prerequisites
- **Python 3.7+**
- **PyTorch** for deep learning models.
- **Transformers Library** by HuggingFace for natural language encoding.

### Installation
1. Clone the repository

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the agent and memory settings by editing `configs/agent_config.yaml` and `configs/memory_config.yaml` to suit your environment.

### Running the Agent
To run the agent in a simulated environment:
```bash
python run_agent.py
```
This will launch the agent, allowing it to explore the environment, remember key features, and respond to natural language commands.
