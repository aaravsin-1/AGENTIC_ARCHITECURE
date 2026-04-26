# 🤖 Complete Agentic AI Cookbook
## LangChain + LangGraph — Every Pattern You Need

---

## 📁 File Index

| File | Pattern | Key Concepts |
|------|---------|-------------|
| `01_basic_react_agent.py` | ReAct Agent | Tools, AgentExecutor, chat history |
| `02_langgraph_state_agent.py` | LangGraph State Machine | StateGraph, nodes, edges, checkpointing |
| `03_supervisor_multiagent.py` | Supervisor Architecture | Manager → workers, structured routing |
| `04_plan_execute_agent.py` | Plan & Execute | Planner, executor, replanner loop |
| `05_reflection_agent.py` | Reflection / Critique | Generator + critic, Constitutional AI |
| `06_multi_agent_network.py` | Parallel Network | Parallel branches, blackboard pattern |
| `07_hierarchical_agents.py` | Hierarchical (Open Claw) | CEO → VP → Worker delegation |
| `08_memory_agent.py` | Long-term Memory | Vector memory, episodic, entity memory |
| `09_agentic_rag.py` | Agentic RAG | CRAG, Self-RAG, adaptive retrieval |
| `10_advanced_tools.py` | Advanced Tools | Async, stateful, dynamic, retry |
| `11_coding_agent.py` | Coding Agent | Write → run → debug → test loop |
| `12_swarm_style_handoffs.py` | Swarm Handoffs | Agent-to-agent transfers, triage |
| `13_production_patterns.py` | Production | Streaming, caching, rate limits, FastAPI |

---

## 🏗️ Architecture Patterns

### 1. ReAct (Reason + Act)
```
User → LLM → Thought → Tool Call → Observation → Thought → ... → Answer
```
**Use when:** Simple tasks, single agent, tool augmentation

---

### 2. LangGraph State Machine
```
START → node_a → node_b → conditional → node_c or node_d → END
              ↑__________________________|
```
**Use when:** Complex control flow, loops, human-in-loop, checkpointing needed

---

### 3. Supervisor Architecture
```
                 Supervisor (LLM router)
                /          |           \
          Researcher     Coder        Writer
               \           |           /
                \__________|__________/
                           ↓
                      Synthesizer
```
**Use when:** Tasks require multiple specialists, clear roles

---

### 4. Plan & Execute
```
User Task → [Planner] → Step 1 → Step 2 → Step 3 → [Replanner] → [Synthesize]
                            ↑___________________________________|
```
**Use when:** Long-horizon tasks, research projects, need upfront planning

---

### 5. Reflection Loop
```
[Generator] → draft → [Critic] → score ≥ 0.85? → [Finalize]
      ↑_____________________________|
```
**Use when:** Quality-critical outputs, writing, code review

---

### 6. Parallel Multi-Agent Network
```
Dispatcher → [Agent A] ─────────────────┐
           → [Agent B] ───────────────── → [Aggregator] → Output
           → [Agent C] ─────────────────┘
```
**Use when:** Different aspects can be analyzed independently and in parallel

---

### 7. Hierarchical (Open Claw)
```
                    CEO
                 /       \
             CTO           CMO
           /   \         /    \
        Dev    QA     Content  Analytics
```
**Use when:** Organization-like tasks, need authority levels, delegation

---

### 8. Swarm Handoffs
```
User → [Triage] --handoff--> [Billing]
                --handoff--> [Technical]
                --handoff--> [Escalation]
```
**Use when:** Customer support, complex routing, agent specialization

---

## 🔑 Core LangGraph Concepts

### State
```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # reducer
    my_field: str                             # plain field
    counter: Annotated[int, operator.add]    # accumulator reducer
```

### Node
```python
def my_node(state: State) -> dict:
    # Read from state
    messages = state["messages"]
    # Return PARTIAL update (only what changed)
    return {"my_field": "new_value", "messages": [new_message]}
```

### Edge Types
```python
# Simple edge
graph.add_edge("node_a", "node_b")

# Conditional edge (routing)
graph.add_conditional_edges(
    "node_a",
    routing_function,          # returns a key
    {"key1": "node_b", "key2": "node_c"}
)

# Parallel edges (fan-out)
graph.add_edge("dispatcher", "agent_1")
graph.add_edge("dispatcher", "agent_2")
graph.add_edge("dispatcher", "agent_3")
# All 3 agents run simultaneously!
```

### Checkpointing (Persistence)
```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver  # file persistence
from langgraph.checkpoint.postgres import PostgresSaver  # production

checkpointer = MemorySaver()
graph = compiled_graph.compile(checkpointer=checkpointer)

# Use thread_id to identify conversations
config = {"configurable": {"thread_id": "user-123-session-1"}}
result = graph.invoke(inputs, config=config)

# Same thread_id → continues the conversation!
result2 = graph.invoke(next_inputs, config=config)
```

### Human-in-the-Loop
```python
# Pause before a node
graph.compile(interrupt_before=["tools"])

# Resume with None (from checkpoint)
graph.invoke(None, config=config)

# Or inject new data
graph.invoke({"messages": [HumanMessage(content="Approved!")]}, config=config)
```

---

## 🛠️ Tool Patterns

### Basic Tool
```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Description — shown to the LLM to decide when to use this."""
    return f"Result: {param}"
```

### Structured Tool (Pydantic)
```python
class MyInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, ge=1, le=20)

@tool(args_schema=MyInput)
def search(query: str, max_results: int = 5) -> str:
    """Search with structured parameters."""
    ...
```

### Tool Node (in LangGraph)
```python
from langgraph.prebuilt import ToolNode

tools = [tool1, tool2, tool3]
tool_node = ToolNode(tools)  # auto-runs any tool_calls in last AI message
```

---

## 💾 Memory Types

| Type | What it stores | Implementation |
|------|---------------|----------------|
| **Buffer** | Last N messages | `MessagesPlaceholder` |
| **Summary** | Compressed history | LLM summarizes old messages |
| **Semantic** | Facts about the world | Vector store (FAISS/Chroma) |
| **Episodic** | Past events/interactions | Vector store with timestamps |
| **Entity** | Facts about specific entities | Dict/knowledge graph |
| **Procedural** | How to do things | Prompt/vector store |

---

## 🚀 Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set real API keys in .env
cp .env.example .env
# Edit .env with real keys

# 3. Run any example
python 01_basic_react_agent.py   # simplest
python 03_supervisor_multiagent.py  # multi-agent
python 07_hierarchical_agents.py    # full hierarchy
python 12_swarm_style_handoffs.py   # swarm

# 4. Deploy
pip install fastapi uvicorn
# Copy FastAPI code from 13_production_patterns.py
uvicorn app:app --reload
```

---

## 🔧 Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...          # or use ANTHROPIC_API_KEY
ANTHROPIC_API_KEY=sk-ant-...

# Optional but recommended
TAVILY_API_KEY=tvly-...        # for web search
LANGCHAIN_API_KEY=lsv2-...     # for LangSmith tracing
LANGCHAIN_TRACING_V2=true      # enable tracing
LANGCHAIN_PROJECT=my-project   # project name in LangSmith
```

---

## 📊 Choosing the Right Pattern

```
Is the task simple with a few tools?
  YES → 01_basic_react_agent.py (ReAct)
  
Does it need complex flow control / loops / human approval?
  YES → 02_langgraph_state_agent.py

Does it need multiple specialists working together?
  YES → Does one agent decide who does what?
    YES → 03_supervisor_multiagent.py
    NO  → Can they work in parallel?
      YES → 06_multi_agent_network.py
      NO  → Is there a hierarchy?
        YES → 07_hierarchical_agents.py
        NO  → 12_swarm_style_handoffs.py

Does it need a plan before executing?
  YES → 04_plan_execute_agent.py

Does quality of output matter a lot?
  YES → 05_reflection_agent.py

Does it need to remember across conversations?
  YES → 08_memory_agent.py

Does it need to search a knowledge base?
  YES → 09_agentic_rag.py

Is it going to production?
  YES → Also apply 13_production_patterns.py patterns
```

---

## 🧰 Key Libraries

```python
# LangChain core
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LLM providers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# Pydantic (structured outputs)
from pydantic import BaseModel, Field

# Utilities
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
```

---

*All files have fake API keys — replace them in `config.py` or via `.env` before running.*
