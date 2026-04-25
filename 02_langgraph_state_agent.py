# ============================================================
# 02_langgraph_state_agent.py
# ─────────────────────────────────────────────────────────────
# LangGraph turns agents into explicit STATE MACHINES.
# Every node is a Python function. Edges decide what runs next.
# This gives you full control, checkpointing, and human-in-loop.
#
# Architecture (graph):
#
#   START → agent_node → (tool calls?) → tool_node → agent_node
#                      ↓ (no tools)
#                     END
# ============================================================

from config import OPENAI_API_KEY, GPT4O_MINI, print_step, console

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages   # reducer: appends messages
from langgraph.prebuilt import ToolNode            # auto-runs tool calls from AI messages
from langgraph.checkpoint.memory import MemorySaver

import json, math, requests


# ── 1. Define the State ────────────────────────────────────────
# State is a TypedDict. Every node receives it and returns updates.
# `add_messages` is a reducer — it APPENDS new messages instead of replacing.

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # You can add any extra fields you want:
    iteration_count: int
    metadata: dict


# ── 2. Define Tools ───────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use for news, facts, recent events."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        output = []
        for r in results:
            output.append(f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}\n")
        return "\n---\n".join(output)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def python_repl(code: str) -> str:
    """
    Execute Python code and return the result.
    Use for calculations, data manipulation, generating content.
    The code runs in a sandboxed environment.
    Always print() results you want to see.
    """
    import io, sys
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        exec(code, {"__builtins__": __builtins__, "math": math, "json": json})
        output = buffer.getvalue()
        return output if output else "Code executed successfully (no output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout


@tool
def read_file(filepath: str) -> str:
    """Read a text file and return its contents. Useful for analyzing documents."""
    try:
        with open(filepath, "r") as f:
            content = f.read()
        return content[:5000]  # truncate large files
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file. Useful for saving results or creating documents."""
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"


# ── 3. Build the Graph ────────────────────────────────────────

def build_agent_graph(tools: list, system_prompt: str = None):
    """
    Build a LangGraph agent graph.
    Returns a compiled graph (callable like a function).
    """
    
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = system_prompt or """You are a powerful AI assistant with access to tools.
Think step by step. Use tools when you need information or need to perform actions.
Always provide clear, accurate, and helpful responses."""
    
    # ── Node 1: The LLM agent ─────────────────────────────────
    def agent_node(state: AgentState) -> dict:
        """The brain: calls the LLM with current messages."""
        messages = state["messages"]
        
        # Inject system prompt at the start
        from langchain_core.messages import SystemMessage
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages
        
        response = llm_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    
    # ── Node 2: Tool executor ─────────────────────────────────
    # ToolNode automatically finds tool_calls in the last AI message,
    # runs each tool, and wraps results in ToolMessage objects.
    tool_node = ToolNode(tools)
    
    # ── Conditional edge: should we call tools or finish? ─────
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Route: if last message has tool calls → tools node, else → end."""
        last_message = state["messages"][-1]
        
        # Safety: stop after too many iterations
        if state.get("iteration_count", 0) > 15:
            return "end"
        
        # If the LLM made tool calls, execute them
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        return "end"
    
    # ── Assemble the graph ────────────────────────────────────
    graph = StateGraph(AgentState)
    
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    
    graph.add_edge(START, "agent")
    
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    
    # After tools run, always go back to agent for reflection
    graph.add_edge("tools", "agent")
    
    # Add checkpointing (persistent memory across invocations)
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    
    return compiled


# ── 4. Visualize the Graph ────────────────────────────────────

def show_graph_structure(graph):
    """Print the graph nodes and edges."""
    console.print("\n[bold cyan]Graph Structure:[/bold cyan]")
    try:
        # Mermaid diagram output
        mermaid = graph.get_graph().draw_mermaid()
        console.print(f"[dim]{mermaid}[/dim]")
    except Exception:
        console.print("[dim]Graph visualization not available[/dim]")


# ── 5. Stream the Agent (real-time output) ────────────────────

def stream_agent(graph, query: str, thread_id: str = "thread-1"):
    """
    Stream agent execution — see each node's output as it happens.
    Uses thread_id for checkpointing (conversation persistence).
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {
        "messages": [HumanMessage(content=query)],
        "iteration_count": 0,
        "metadata": {},
    }
    
    console.print(f"\n[bold white]USER:[/bold white] {query}\n")
    
    final_answer = None
    
    # stream_mode="updates" emits each node's state update
    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "agent":
                last_msg = node_output["messages"][-1]
                
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        console.print(f"  [bold yellow]🔧 Calling tool:[/bold yellow] {tc['name']}({json.dumps(tc['args'])[:80]})")
                else:
                    final_answer = last_msg.content
                    
            elif node_name == "tools":
                for msg in node_output["messages"]:
                    if isinstance(msg, ToolMessage):
                        console.print(f"  [bold blue]📋 Tool result ({msg.name}):[/bold blue] {str(msg.content)[:120]}...")
    
    if final_answer:
        console.print(f"\n[bold green]AGENT:[/bold green] {final_answer}")
    
    return final_answer


# ── 6. Human-in-the-Loop Pattern ─────────────────────────────

def build_agent_with_interrupt(tools: list):
    """
    Build an agent that PAUSES before executing tools,
    allowing a human to approve or reject each tool call.
    This is the 'human-in-the-loop' pattern.
    """
    from langgraph.graph import StateGraph
    
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response], "iteration_count": state.get("iteration_count", 0) + 1}
    
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"
    
    tool_node = ToolNode(tools)
    
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    
    checkpointer = MemorySaver()
    
    # ↓↓↓ THE KEY: interrupt_before=["tools"] pauses before tool execution
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],   # pause here for human approval
    )
    return compiled


def run_with_human_approval(query: str):
    """Demo of human-in-the-loop approval workflow."""
    tools = [web_search, python_repl]
    graph = build_agent_with_interrupt(tools)
    
    config = {"configurable": {"thread_id": "human-loop-demo"}}
    inputs = {
        "messages": [HumanMessage(content=query)],
        "iteration_count": 0,
        "metadata": {},
    }
    
    console.print(f"\n[bold white]USER:[/bold white] {query}")
    
    # First run — will stop before tool execution
    result = graph.invoke(inputs, config=config)
    last_msg = result["messages"][-1]
    
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            console.print(f"\n[bold yellow]⚠️  Agent wants to call:[/bold yellow] {tc['name']}")
            console.print(f"[dim]   Args: {tc['args']}[/dim]")
            
            # In a real app, you'd show this to the user via UI
            approval = input("   Approve? (y/n): ").strip().lower()
            
            if approval != "y":
                # Inject a rejection message and continue without the tool
                result["messages"].append(
                    HumanMessage(content=f"Do not use the {tc['name']} tool. Answer without it.")
                )
    
    # Resume the graph (continues from where it paused)
    final = graph.invoke(None, config=config)  # None = resume from checkpoint
    console.print(f"\n[bold green]AGENT:[/bold green] {final['messages'][-1].content}")


# ── 7. Main demo ──────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold magenta]═══ 02: LangGraph State Machine Agent ═══[/bold magenta]")
    
    tools = [web_search, python_repl, write_file]
    graph = build_agent_graph(tools)
    
    show_graph_structure(graph)
    
    # Demo 1: Basic query
    stream_agent(graph, "Calculate the first 10 Fibonacci numbers using Python.")
    
    # Demo 2: Persistent memory — follow-up question in same thread
    stream_agent(graph, "Now square each of those numbers.", thread_id="thread-1")
    
    # Demo 3: New thread (fresh context)
    stream_agent(
        graph,
        "Write a Python function to check if a number is prime, test it with 17 and 20, and save the code to /tmp/prime_checker.py",
        thread_id="thread-2"
    )
    
    # Demo 4: Human-in-the-loop
    # run_with_human_approval("Search the web for the latest AI news")
