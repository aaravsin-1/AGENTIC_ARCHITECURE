# ============================================================
# 01_basic_react_agent.py
# ─────────────────────────────────────────────────────────────
# The ReAct (Reason + Act) pattern: the simplest agent loop.
# The LLM alternates between Thought → Action → Observation
# until it decides it has a final answer.
#
# Architecture:
#   User ──► Agent (LLM + tools) ──► Tool calls ──► Final answer
# ============================================================

from config import OPENAI_API_KEY, GPT4O_MINI, print_step, console
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import math, json, requests
from datetime import datetime

# ── 1. Define Tools ───────────────────────────────────────────
# Tools are just Python functions decorated with @tool.
# The docstring becomes the tool's description for the LLM.

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Supports: +, -, *, /, **, sqrt, sin, cos, log, etc.
    Example input: "sqrt(144) + 2**10"
    """
    try:
        # Safe eval with only math functions
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "pow": pow})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    Get the current date and time.
    Args:
        timezone: Timezone name (e.g., 'UTC', 'US/Eastern'). Default is UTC.
    """
    now = datetime.utcnow()
    return f"Current time (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia and return a summary of the topic.
    Use this to look up facts, history, science, people, etc.
    Args:
        query: The search term to look up on Wikipedia.
    """
    try:
        import wikipedia
        result = wikipedia.summary(query, sentences=3)
        return result
    except Exception as e:
        return f"Wikipedia search failed: {e}. Try a more specific query."


@tool
def get_weather(city: str) -> str:
    """
    Get current weather information for a city.
    Args:
        city: Name of the city (e.g., 'London', 'New York', 'Tokyo')
    """
    # Using free wttr.in API — no key needed
    try:
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        current = data["current_condition"][0]
        return (
            f"Weather in {city}: "
            f"{current['weatherDesc'][0]['value']}, "
            f"Temp: {current['temp_C']}°C / {current['temp_F']}°F, "
            f"Humidity: {current['humidity']}%, "
            f"Wind: {current['windspeedKmph']} km/h"
        )
    except Exception as e:
        return f"Could not fetch weather for {city}: {e}"


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units of measurement.
    Supports: km/miles, kg/lbs, celsius/fahrenheit, liters/gallons
    Args:
        value: The numeric value to convert
        from_unit: The source unit (e.g., 'km', 'kg', 'celsius', 'liters')
        to_unit: The target unit (e.g., 'miles', 'lbs', 'fahrenheit', 'gallons')
    """
    conversions = {
        ("km", "miles"):       lambda v: v * 0.621371,
        ("miles", "km"):       lambda v: v * 1.60934,
        ("kg", "lbs"):         lambda v: v * 2.20462,
        ("lbs", "kg"):         lambda v: v * 0.453592,
        ("celsius", "fahrenheit"): lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5/9,
        ("liters", "gallons"): lambda v: v * 0.264172,
        ("gallons", "liters"): lambda v: v * 3.78541,
        ("meters", "feet"):    lambda v: v * 3.28084,
        ("feet", "meters"):    lambda v: v * 0.3048,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"Conversion from {from_unit} to {to_unit} not supported."


# ── 2. Build the Agent ────────────────────────────────────────
def create_react_agent():
    """Create a ReAct agent with all tools."""
    
    llm = ChatOpenAI(
        model=GPT4O_MINI,
        temperature=0,          # deterministic for tool calls
        api_key=OPENAI_API_KEY,
    )
    
    tools = [calculator, get_current_time, search_wikipedia, get_weather, unit_converter]
    
    # Prompt structure: system → history → user input → agent scratchpad
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to tools.
        
When answering questions:
1. Think about what tools you need
2. Use tools to gather accurate information
3. Synthesize the results into a clear, helpful answer
4. Always be honest if you don't know something

Available tools: calculator, get_current_time, search_wikipedia, get_weather, unit_converter"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # where tool calls go
    ])
    
    # create_tool_calling_agent uses the model's native function-calling
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # AgentExecutor runs the loop: agent → tool → agent → ... → final answer
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,           # prints each step
        max_iterations=10,      # safety limit
        handle_parsing_errors=True,
        return_intermediate_steps=True,  # gives you full trace
    )
    
    return executor


# ── 3. Run Examples ───────────────────────────────────────────
def run_examples():
    console.print("\n[bold magenta]═══ 01: Basic ReAct Agent ═══[/bold magenta]\n")
    
    agent = create_react_agent()
    
    test_questions = [
        "What is the square root of 1764 plus 2 to the power of 8?",
        "How many miles is 42.195 km (a marathon)?",
        "Who invented the telephone and when?",
        "What's the weather like in Tokyo right now?",
        "What time is it in UTC, and what day of the week is that?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print_step(f"Question {i}", question, "cyan")
        
        result = agent.invoke({"input": question})
        
        print_step(f"Answer {i}", result["output"], "green")
        
        # Show intermediate steps (tool calls)
        if result.get("intermediate_steps"):
            steps_summary = []
            for action, observation in result["intermediate_steps"]:
                steps_summary.append(f"→ Tool: {action.tool}({action.tool_input}) → {str(observation)[:100]}")
            console.print("\n[dim]Tool calls made:[/dim]")
            for s in steps_summary:
                console.print(f"  [dim]{s}[/dim]")
        
        console.print("\n" + "─" * 60)


# ── 4. Multi-turn conversation with memory ────────────────────
def run_with_memory():
    """Show how to maintain conversation context."""
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    
    console.print("\n[bold magenta]═══ ReAct Agent with Chat History ═══[/bold magenta]\n")
    
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    tools = [calculator, search_wikipedia, get_weather]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use tools when needed. Remember the conversation history."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    # Wrap with message history
    store = {}
    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    agent_with_history = RunnableWithMessageHistory(
        executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    session = {"configurable": {"session_id": "demo-session-1"}}
    
    # Multi-turn conversation
    turns = [
        "My name is Alex and I'm planning a trip to Paris.",
        "What's the weather like there?",
        "How many miles is it from New York to Paris? (5,836 km)",
        "Based on everything we discussed, give me a quick trip summary.",
    ]
    
    for msg in turns:
        console.print(f"\n[bold white]USER:[/bold white] {msg}")
        result = agent_with_history.invoke({"input": msg}, config=session)
        console.print(f"[bold green]AGENT:[/bold green] {result['output']}")


if __name__ == "__main__":
    run_examples()
    run_with_memory()
