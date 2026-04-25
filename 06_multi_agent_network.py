# ============================================================
# 06_multi_agent_network.py
# ─────────────────────────────────────────────────────────────
# MULTI-AGENT NETWORK — "Open Claw" / fully connected architecture.
# Agents run in PARALLEL branches and share a global state.
# Results are merged and synthesized.
#
# Architecture:
#
#   ┌─────────────┐
#   │  DISPATCHER │  ← analyzes task, decides parallel branches
#   └──────┬──────┘
#     ┌────┼────┐────────┐
#  ┌──▼─┐ ┌▼──┐ ┌▼──┐  ┌▼──┐
#  │ A  │ │ B │ │ C │  │ D │  ← agents run simultaneously
#  └──┬─┘ └┬──┘ └┬──┘  └┬──┘
#     └────┴────┴────────┘
#          ↓
#    ┌───────────┐
#    │ AGGREGATOR│  ← merges all parallel outputs
#    └─────┬─────┘
#          ↓
#      FINAL OUTPUT
# ============================================================

import asyncio
from config import OPENAI_API_KEY, GPT4O_MINI, console, print_step

from typing import Annotated, Dict, List, Optional, Any
from typing_extensions import TypedDict
import operator

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
import json, time


# ── 1. State with parallel branches ───────────────────────────
# `operator.add` as reducer means lists are CONCATENATED (not replaced)

class NetworkState(TypedDict):
    task: str
    branch_tasks: Dict[str, str]               # branch_name → sub-task
    branch_outputs: Annotated[Dict[str, str], lambda a, b: {**a, **b}]  # merge dicts
    active_branches: List[str]
    messages: Annotated[list, add_messages]
    synthesis: str
    metadata: Dict[str, Any]


# ── 2. Agent factory (creates any specialist agent) ───────────

def create_specialist(
    name: str,
    system_prompt: str,
    tools: list = None,
    temperature: float = 0.3,
) -> callable:
    """Factory: create any specialist agent node."""
    
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=temperature, api_key=OPENAI_API_KEY)
    
    if tools:
        llm_with_tools = llm.bind_tools(tools)
        tool_node = ToolNode(tools)
    else:
        llm_with_tools = llm
        tool_node = None
    
    def agent_fn(state: NetworkState) -> dict:
        # Get this agent's specific sub-task
        sub_task = state["branch_tasks"].get(name, state["task"])
        
        start_time = time.time()
        console.print(f"  [bold]🤖 {name.upper()}[/bold] starting...")
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Your task: {sub_task}\n\nOverall context: {state['task']}")
        ]
        
        if tool_node:
            # Agent with tools — run mini loop
            for _ in range(5):
                response = llm_with_tools.invoke(messages)
                messages.append(response)
                
                if not (hasattr(response, "tool_calls") and response.tool_calls):
                    break
                
                tool_results = tool_node.invoke({"messages": messages})
                messages.extend(tool_results["messages"])
            
            output = messages[-1].content
        else:
            response = llm.invoke(messages)
            output = response.content
        
        elapsed = time.time() - start_time
        console.print(f"  [green]✓ {name.upper()}[/green] completed in {elapsed:.1f}s")
        
        return {
            "branch_outputs": {name: output},
            "messages": [AIMessage(content=f"[{name.upper()}]: {output[:80]}...")],
        }
    
    agent_fn.__name__ = name
    return agent_fn


# ── 3. Tools for specialists ──────────────────────────────────

@tool
def search(query: str) -> str:
    """Search for information."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return "\n".join(f"{r['title']}: {r['body'][:150]}" for r in results)
    except:
        return f"Mock search result for: {query}"


@tool
def run_code(code: str) -> str:
    """Execute Python code."""
    import io, sys
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        exec(code, {"__builtins__": __builtins__})
        return buf.getvalue() or "OK"
    except Exception as e:
        return f"Error: {e}"
    finally:
        sys.stdout = old_stdout


@tool
def analyze_data(data: str) -> str:
    """Analyze structured data (JSON or CSV-like)."""
    import statistics
    try:
        # Try to parse as JSON
        parsed = json.loads(data)
        if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
            return json.dumps({
                "count": len(parsed),
                "mean": statistics.mean(parsed),
                "median": statistics.median(parsed),
                "stdev": statistics.stdev(parsed) if len(parsed) > 1 else 0,
                "min": min(parsed),
                "max": max(parsed),
            }, indent=2)
        return f"Data structure: {type(parsed).__name__} with {len(parsed)} items"
    except Exception as e:
        return f"Analysis: {data[:200]}"


# ── 4. Build the Parallel Network Graph ───────────────────────

class BranchPlan(BaseModel):
    """How to decompose the task across parallel branches."""
    research_task: str = Field(description="Task for the research agent")
    analysis_task: str = Field(description="Task for the data analysis agent")
    creative_task: str = Field(description="Task for the creative writing agent")
    code_task: str = Field(description="Task for the coding agent")
    critique_task: str = Field(description="Task for the critical thinking agent")


def build_network_graph():
    
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    
    # ── Dispatcher: decomposes task into parallel branches ────
    def dispatcher_node(state: NetworkState) -> dict:
        console.print(f"\n[bold magenta]📡 DISPATCHER: Decomposing task into branches...[/bold magenta]")
        
        structured_llm = llm.with_structured_output(BranchPlan)
        
        plan = structured_llm.invoke([
            SystemMessage(content="""Decompose the user's task into specialized sub-tasks 
for different expert agents to work on in PARALLEL.
Each agent should have a distinct, complementary role.
Make each sub-task specific and actionable."""),
            HumanMessage(content=f"Decompose this task: {state['task']}"),
        ])
        
        branch_tasks = {
            "researcher":  plan.research_task,
            "analyst":     plan.analysis_task,
            "creative":    plan.creative_task,
            "coder":       plan.code_task,
            "critic":      plan.critique_task,
        }
        
        console.print(f"  [dim]Branches: {list(branch_tasks.keys())}[/dim]")
        for branch, task in branch_tasks.items():
            console.print(f"  [cyan]{branch}[/cyan]: {task[:80]}...")
        
        return {
            "branch_tasks": branch_tasks,
            "active_branches": list(branch_tasks.keys()),
        }
    
    # ── Create specialist agents ──────────────────────────────
    researcher = create_specialist(
        "researcher",
        """You are a research expert. Find and synthesize information from multiple angles.
Focus on facts, evidence, and authoritative sources.
Structure your findings clearly with key insights.""",
        tools=[search],
        temperature=0.1,
    )
    
    analyst = create_specialist(
        "analyst",
        """You are a data and systems analyst. 
Examine patterns, structures, implications, and tradeoffs.
Think quantitatively when possible. Use numbers and metrics.""",
        tools=[analyze_data, run_code],
        temperature=0.1,
    )
    
    creative = create_specialist(
        "creative",
        """You are a creative director. Generate novel ideas, metaphors, frameworks,
and unexpected connections. Think divergently and imaginatively.
Produce content that is memorable and engaging.""",
        temperature=0.9,  # high temp for creativity
    )
    
    coder = create_specialist(
        "coder",
        """You are a software engineer. Write working, clean, well-documented code.
Solve problems with elegant implementations.
Test and explain your code clearly.""",
        tools=[run_code],
        temperature=0.0,
    )
    
    critic = create_specialist(
        "critic",
        """You are a critical thinking expert. Identify flaws, risks, edge cases,
and weaknesses in ideas and approaches.
Provide balanced critique with constructive alternatives.""",
        temperature=0.2,
    )
    
    # ── Aggregator: merges all parallel outputs ───────────────
    def aggregator_node(state: NetworkState) -> dict:
        console.print(f"\n[bold blue]🔀 AGGREGATOR: Merging {len(state['branch_outputs'])} branch outputs...[/bold blue]")
        
        # Build a structured summary of all outputs
        outputs_text = ""
        for branch_name, output in state["branch_outputs"].items():
            outputs_text += f"\n\n### {branch_name.upper()}\n{output}"
        
        aggregator_prompt = f"""You've received outputs from multiple specialist agents.
Original task: {state['task']}

Agent outputs:
{outputs_text}

Create a comprehensive, unified synthesis that:
1. Integrates insights from all agents
2. Resolves any contradictions
3. Highlights the most important findings
4. Provides a clear, actionable final answer

Format with clear sections."""
        
        response = llm.invoke([
            SystemMessage(content="You are an expert synthesizer. Create a masterful integration of multiple expert perspectives."),
            HumanMessage(content=aggregator_prompt),
        ])
        
        return {
            "synthesis": response.content,
            "messages": [AIMessage(content=f"[AGGREGATOR]: Synthesis complete")],
        }
    
    # ── Build the graph ───────────────────────────────────────
    graph = StateGraph(NetworkState)
    
    # Add all nodes
    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("researcher", researcher)
    graph.add_node("analyst",    analyst)
    graph.add_node("creative",   creative)
    graph.add_node("coder",      coder)
    graph.add_node("critic",     critic)
    graph.add_node("aggregator", aggregator_node)
    
    # Dispatcher → all parallel agents simultaneously
    graph.add_edge(START, "dispatcher")
    graph.add_edge("dispatcher", "researcher")
    graph.add_edge("dispatcher", "analyst")
    graph.add_edge("dispatcher", "creative")
    graph.add_edge("dispatcher", "coder")
    graph.add_edge("dispatcher", "critic")
    
    # All agents → aggregator
    graph.add_edge("researcher", "aggregator")
    graph.add_edge("analyst",    "aggregator")
    graph.add_edge("creative",   "aggregator")
    graph.add_edge("coder",      "aggregator")
    graph.add_edge("critic",     "aggregator")
    
    graph.add_edge("aggregator", END)
    
    return graph.compile()


# ── 5. Async Parallel Execution ───────────────────────────────
# For true parallelism, run branches as async coroutines

async def run_parallel_agents(tasks: Dict[str, str]) -> Dict[str, str]:
    """Run multiple agent tasks in true async parallel."""
    
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0.3, api_key=OPENAI_API_KEY)
    
    async def run_single_agent(name: str, task: str) -> tuple[str, str]:
        response = await llm.ainvoke([
            SystemMessage(content=f"You are the {name} specialist agent."),
            HumanMessage(content=task),
        ])
        return name, response.content
    
    # Launch all tasks concurrently
    coroutines = [run_single_agent(name, task) for name, task in tasks.items()]
    results = await asyncio.gather(*coroutines)
    
    return dict(results)


# ── 6. Agent Communication Protocol ──────────────────────────
# Agents can send messages to each other (blackboard pattern)

class AgentMessage(BaseModel):
    """A message between agents."""
    from_agent: str
    to_agent: str          # "all" for broadcast
    message_type: str      # "result", "question", "clarification", "alert"
    content: str
    priority: int = 1      # 1=low, 5=critical


class BlackboardNetwork:
    """
    Agents share a common 'blackboard' (shared memory).
    Any agent can read/write to the blackboard.
    This enables emergent collaboration.
    """
    
    def __init__(self):
        self.blackboard: Dict[str, Any] = {}
        self.messages: List[AgentMessage] = []
        self.llm = ChatOpenAI(model=GPT4O_MINI, temperature=0.2, api_key=OPENAI_API_KEY)
    
    def post_result(self, agent: str, key: str, value: Any):
        """An agent posts a result to the shared blackboard."""
        self.blackboard[f"{agent}:{key}"] = value
        console.print(f"  [dim]{agent} → blackboard[{key}][/dim]")
    
    def read_all(self) -> str:
        """Read all blackboard entries as context."""
        if not self.blackboard:
            return "Blackboard is empty."
        return "\n".join(f"{k}: {str(v)[:200]}" for k, v in self.blackboard.items())
    
    def send_message(self, msg: AgentMessage):
        """Send a message to another agent."""
        self.messages.append(msg)
        console.print(f"  [dim]{msg.from_agent} → {msg.to_agent}: [{msg.message_type}][/dim]")
    
    def get_messages_for(self, agent: str) -> List[AgentMessage]:
        """Get all messages addressed to a specific agent."""
        return [m for m in self.messages if m.to_agent in (agent, "all")]
    
    def run_collaborative_task(self, task: str) -> str:
        """Run a task with blackboard-based agent collaboration."""
        console.print(f"\n[bold cyan]Blackboard Network:[/bold cyan]")
        console.print(f"Task: {task}\n")
        
        agents = ["researcher", "analyst", "synthesizer"]
        
        for agent in agents:
            # Read current blackboard
            context = self.read_all()
            
            # Get messages directed to this agent
            my_messages = self.get_messages_for(agent)
            message_context = "\n".join(f"From {m.from_agent}: {m.content}" for m in my_messages)
            
            prompt = f"""You are the {agent} agent in a collaborative network.

Task: {task}

Current shared knowledge (blackboard):
{context if context != "Blackboard is empty." else "Nothing yet — you're going first."}

{"Messages for you:" + message_context if message_context else ""}

Do your part of the work. Post your key findings and any questions for other agents."""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Post results back to blackboard
            self.post_result(agent, "analysis", response.content[:500])
            
            console.print(f"  [bold]{agent.upper()}[/bold]: {response.content[:100]}...")
        
        # Final synthesis using all blackboard content
        final = self.llm.invoke([
            SystemMessage(content="Synthesize all collaborative work into a final answer."),
            HumanMessage(content=f"Task: {task}\n\nAll agent outputs:\n{self.read_all()}"),
        ])
        
        return final.content


# ── 7. Main Demo ──────────────────────────────────────────────

def run_network_demo(task: str):
    console.print(f"\n[bold magenta]═══ 06: Multi-Agent Network ═══[/bold magenta]")
    console.print(f"\n[bold white]TASK:[/bold white] {task}\n")
    
    start = time.time()
    graph = build_network_graph()
    
    initial = {
        "task": task,
        "branch_tasks": {},
        "branch_outputs": {},
        "active_branches": [],
        "messages": [HumanMessage(content=task)],
        "synthesis": "",
        "metadata": {},
    }
    
    result = graph.invoke(initial)
    elapsed = time.time() - start
    
    print_step(
        f"Network Output ({elapsed:.1f}s, {len(result['branch_outputs'])} agents)",
        result["synthesis"][:1000] + "..."
    )
    
    return result


if __name__ == "__main__":
    # Demo 1: Full parallel network
    run_network_demo(
        "Design and analyze a system for a social media platform that uses AI "
        "to recommend content, detect misinformation, and personalize feeds."
    )
    
    # Demo 2: Async parallel (run multiple tasks truly in parallel)
    async def async_demo():
        tasks = {
            "economist":   "What are the pros and cons of a 4-day work week economically?",
            "psychologist": "How does a 4-day work week affect mental health and productivity?",
            "manager":     "What operational challenges does a 4-day work week create?",
        }
        console.print("\n[bold cyan]Async Parallel Demo:[/bold cyan]")
        results = await run_parallel_agents(tasks)
        for agent, result in results.items():
            console.print(f"\n[bold]{agent.upper()}:[/bold] {result[:200]}...")
    
    asyncio.run(async_demo())
    
    # Demo 3: Blackboard network
    blackboard = BlackboardNetwork()
    result = blackboard.run_collaborative_task(
        "Evaluate whether Python or Rust is better for building high-performance web APIs."
    )
    print_step("Blackboard Network Output", result[:800])
