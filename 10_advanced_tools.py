# ============================================================
# 10_advanced_tools.py
# ─────────────────────────────────────────────────────────────
# ADVANCED TOOL PATTERNS
# Everything about building great tools for agents:
#
# A) Structured input/output tools (Pydantic schemas)
# B) Async tools (parallel execution)
# C) Tools with state/side-effects
# D) Tool chaining (tools that call tools)
# E) Dynamic tool generation
# F) Tool error handling and retry
# G) Human-confirmation tools
# H) API integration tools (REST, databases)
# ============================================================

from config import OPENAI_API_KEY, GPT4O_MINI, console, print_step

import asyncio, json, time, uuid
from typing import Any, Dict, List, Optional, Type
from typing_extensions import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool, StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
# A) STRUCTURED INPUT TOOLS (Pydantic schemas)
# These enforce types, validation, and documentation.
# ══════════════════════════════════════════════════════════════

class EmailInput(BaseModel):
    """Input schema for send_email tool."""
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    cc: Optional[List[str]] = Field(default=None, description="CC recipients")
    html: bool = Field(default=False, description="Send as HTML email")


class EmailOutput(BaseModel):
    """Output schema for send_email tool."""
    message_id: str
    status: str
    sent_at: str


@tool(args_schema=EmailInput)
def send_email(to: str, subject: str, body: str, cc: Optional[List[str]] = None, html: bool = False) -> str:
    """
    Send an email to a recipient.
    Use for notifications, reports, or any communication tasks.
    """
    # In production: use smtplib, sendgrid, AWS SES, etc.
    msg_id = str(uuid.uuid4())[:8]
    console.print(f"  [dim]📧 Email sent to {to}: '{subject}' (id: {msg_id})[/dim]")
    
    result = EmailOutput(
        message_id=msg_id,
        status="sent",
        sent_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return result.model_dump_json()


class DatabaseQueryInput(BaseModel):
    """Input for database queries."""
    query: str = Field(description="SQL SELECT query to execute")
    database: str = Field(default="main", description="Database name: 'main', 'analytics', 'logs'")
    limit: int = Field(default=100, ge=1, le=1000, description="Max rows to return")


@tool(args_schema=DatabaseQueryInput)
def query_database(query: str, database: str = "main", limit: int = 100) -> str:
    """
    Execute a SQL query against the database.
    Only SELECT queries are allowed. Returns results as JSON.
    Available databases: main (user data), analytics (metrics), logs (events).
    """
    # Mock database with sample data
    mock_data = {
        "main": [
            {"id": 1, "name": "Alice", "email": "alice@example.com", "plan": "pro", "revenue": 99},
            {"id": 2, "name": "Bob",   "email": "bob@example.com",   "plan": "basic", "revenue": 29},
            {"id": 3, "name": "Carol", "email": "carol@example.com", "plan": "pro", "revenue": 99},
            {"id": 4, "name": "Dave",  "email": "dave@example.com",  "plan": "enterprise", "revenue": 499},
        ],
        "analytics": [
            {"date": "2024-01-01", "dau": 1200, "signups": 45, "revenue": 8920},
            {"date": "2024-01-02", "dau": 1350, "signups": 52, "revenue": 9540},
            {"date": "2024-01-03", "dau": 1180, "signups": 38, "revenue": 8100},
        ],
        "logs": [
            {"timestamp": "2024-01-03T10:00:00Z", "level": "ERROR", "message": "Database timeout"},
            {"timestamp": "2024-01-03T10:05:00Z", "level": "INFO",  "message": "Retry succeeded"},
        ]
    }
    
    if not query.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries are allowed."})
    
    data = mock_data.get(database, [])[:limit]
    return json.dumps({"rows": data, "count": len(data), "database": database})


# ══════════════════════════════════════════════════════════════
# B) ASYNC TOOLS (for parallel execution)
# ══════════════════════════════════════════════════════════════

@tool
async def fetch_api_async(url: str, method: str = "GET", payload: str = "") -> str:
    """
    Make an async HTTP request to an API endpoint.
    Runs without blocking other operations.
    Args:
        url: The API endpoint URL
        method: HTTP method (GET, POST, PUT, DELETE)
        payload: JSON payload for POST/PUT requests
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            if method.upper() == "GET":
                resp = await client.get(url)
            elif method.upper() == "POST":
                data = json.loads(payload) if payload else {}
                resp = await client.post(url, json=data)
            else:
                return f"Method {method} not supported in demo."
            
            return json.dumps({
                "status_code": resp.status_code,
                "body": resp.text[:500],
            })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def run_tools_in_parallel(tasks: List[Dict]) -> List[str]:
    """Run multiple tool calls truly in parallel."""
    
    async def run_one(task: Dict) -> str:
        tool_name = task["tool"]
        args = task["args"]
        
        # Simulate async work
        await asyncio.sleep(0.1)
        
        if tool_name == "fetch_api":
            return await fetch_api_async.arun(args)
        return f"Result for {tool_name}: {args}"
    
    return await asyncio.gather(*[run_one(t) for t in tasks])


# ══════════════════════════════════════════════════════════════
# C) STATEFUL TOOLS (tools with persistent state)
# ══════════════════════════════════════════════════════════════

class ShoppingCart:
    """A stateful shopping cart tool group."""
    
    def __init__(self):
        self.items: Dict[str, Dict] = {}
        self.discount_code: Optional[str] = None
    
    def get_add_to_cart_tool(self):
        cart = self  # capture reference
        
        @tool
        def add_to_cart(product_name: str, quantity: int, price: float) -> str:
            """Add a product to the shopping cart."""
            if product_name in cart.items:
                cart.items[product_name]["quantity"] += quantity
            else:
                cart.items[product_name] = {"quantity": quantity, "price": price}
            total = sum(v["quantity"] * v["price"] for v in cart.items.values())
            return f"Added {quantity}x {product_name} @ ${price}. Cart total: ${total:.2f}"
        
        return add_to_cart
    
    def get_view_cart_tool(self):
        cart = self
        
        @tool
        def view_cart() -> str:
            """View the current contents of the shopping cart."""
            if not cart.items:
                return "Cart is empty."
            
            lines = ["Shopping Cart:"]
            total = 0
            for name, data in cart.items.items():
                subtotal = data["quantity"] * data["price"]
                total += subtotal
                lines.append(f"  {data['quantity']}x {name} @ ${data['price']:.2f} = ${subtotal:.2f}")
            
            if cart.discount_code:
                discount = total * 0.1
                lines.append(f"  Discount (10%): -${discount:.2f}")
                total -= discount
            
            lines.append(f"  TOTAL: ${total:.2f}")
            return "\n".join(lines)
        
        return view_cart
    
    def get_apply_discount_tool(self):
        cart = self
        
        @tool
        def apply_discount(code: str) -> str:
            """Apply a discount code to the cart."""
            valid_codes = {"SAVE10": 10, "SUMMER20": 20, "VIP30": 30}
            if code.upper() in valid_codes:
                cart.discount_code = code.upper()
                pct = valid_codes[code.upper()]
                return f"Discount code '{code}' applied! {pct}% off your order."
            return f"Invalid discount code: '{code}'. Valid codes: {', '.join(valid_codes.keys())}"
        
        return apply_discount
    
    def get_all_tools(self):
        return [
            self.get_add_to_cart_tool(),
            self.get_view_cart_tool(),
            self.get_apply_discount_tool(),
        ]


# ══════════════════════════════════════════════════════════════
# D) TOOL CHAINING (tools that call other tools / LLMs)
# ══════════════════════════════════════════════════════════════

@tool
def summarize_and_translate(text: str, target_language: str) -> str:
    """
    Summarize a long text AND translate it to the target language.
    Internally chains: summarize → translate.
    Args:
        text: The input text to process
        target_language: Target language (e.g., 'Spanish', 'French', 'Japanese')
    """
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    
    # Step 1: Summarize
    summary = llm.invoke([
        SystemMessage(content="Summarize the text in 2-3 sentences. Be concise."),
        HumanMessage(content=text),
    ])
    
    # Step 2: Translate the summary
    translated = llm.invoke([
        SystemMessage(content=f"Translate the following text to {target_language}. Output only the translation."),
        HumanMessage(content=summary.content),
    ])
    
    return f"Summary (English): {summary.content}\n\nTranslation ({target_language}): {translated.content}"


@tool
def research_and_cite(topic: str) -> str:
    """
    Research a topic and generate a properly cited response.
    Combines web search with citation formatting.
    Args:
        topic: The topic to research
    """
    # In production this would do real web search + citation
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0.1, api_key=OPENAI_API_KEY)
    
    # Simulate research
    research = llm.invoke([
        SystemMessage(content="Generate a researched response with 2-3 mock citations in [Author, Year] format."),
        HumanMessage(content=f"Research this topic: {topic}"),
    ])
    
    return research.content


# ══════════════════════════════════════════════════════════════
# E) DYNAMIC TOOL GENERATION
# Generate tools at runtime based on config/user preferences
# ══════════════════════════════════════════════════════════════

def create_api_tool(
    name: str,
    description: str,
    base_url: str,
    endpoint: str,
    method: str = "GET",
    api_key_header: str = None,
    api_key: str = None,
) -> BaseTool:
    """
    Dynamically generate an API tool from a configuration.
    Useful for creating tools for any REST API at runtime.
    """
    import requests
    
    class DynamicAPIInput(BaseModel):
        query_params: Optional[str] = Field(
            default=None,
            description="Query parameters as JSON string"
        )
        body: Optional[str] = Field(
            default=None,
            description="Request body as JSON string"
        )
    
    def api_call(query_params: Optional[str] = None, body: Optional[str] = None) -> str:
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {}
        if api_key_header and api_key:
            headers[api_key_header] = api_key
        
        params = json.loads(query_params) if query_params else {}
        payload = json.loads(body) if body else {}
        
        try:
            if method.upper() == "GET":
                resp = requests.get(url, headers=headers, params=params, timeout=10)
            else:
                resp = requests.request(method, url, headers=headers, json=payload, timeout=10)
            return resp.text[:1000]
        except Exception as e:
            return f"API call failed: {e}"
    
    return StructuredTool.from_function(
        func=api_call,
        name=name,
        description=description,
        args_schema=DynamicAPIInput,
    )


# Generate tools from a config file/dict
def generate_tools_from_config(tool_configs: List[Dict]) -> List[BaseTool]:
    """Create multiple API tools from a configuration list."""
    tools = []
    for config in tool_configs:
        tool_obj = create_api_tool(
            name=config["name"],
            description=config["description"],
            base_url=config["base_url"],
            endpoint=config.get("endpoint", "/"),
            method=config.get("method", "GET"),
            api_key_header=config.get("api_key_header"),
            api_key=config.get("api_key"),
        )
        tools.append(tool_obj)
    return tools


# ══════════════════════════════════════════════════════════════
# F) TOOL ERROR HANDLING & RETRY
# ══════════════════════════════════════════════════════════════

from functools import wraps

def with_retry(max_retries: int = 3, backoff: float = 1.0):
    """Decorator: retry a tool on failure with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        return f"Tool failed after {max_retries} attempts: {e}"
                    wait = backoff * (2 ** attempt)
                    console.print(f"  [yellow]⚠ Tool error (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s...[/yellow]")
                    time.sleep(wait)
        return wrapper
    return decorator


@tool
@with_retry(max_retries=3)
def flaky_external_api(endpoint: str) -> str:
    """
    Call an external API (simulated with random failures).
    This tool auto-retries on failure.
    """
    import random
    if random.random() < 0.5:  # 50% chance of failure
        raise Exception("Connection timeout")
    return f"Success from {endpoint}: {{'data': 'response_data', 'status': 'ok'}}"


class ToolWithFallback:
    """
    A tool that has a fallback strategy if the primary method fails.
    """
    
    def __init__(self, primary_tool, fallback_tool, llm: ChatOpenAI):
        self.primary = primary_tool
        self.fallback = fallback_tool
        self.llm = llm
    
    def run(self, input_str: str) -> str:
        try:
            return self.primary.run(input_str)
        except Exception as primary_error:
            console.print(f"  [yellow]Primary tool failed: {primary_error}. Trying fallback...[/yellow]")
            try:
                return self.fallback.run(input_str)
            except Exception as fallback_error:
                # Final fallback: use LLM knowledge
                response = self.llm.invoke([
                    SystemMessage(content="The tools failed. Use your own knowledge to answer."),
                    HumanMessage(content=input_str),
                ])
                return f"[From LLM knowledge]\n{response.content}"


# ══════════════════════════════════════════════════════════════
# G) HUMAN CONFIRMATION TOOL
# Some actions require explicit human approval
# ══════════════════════════════════════════════════════════════

class PendingActions:
    """Store for pending actions awaiting human approval."""
    def __init__(self):
        self.pending: Dict[str, Dict] = {}
    
    def add(self, action: str, details: Dict) -> str:
        action_id = str(uuid.uuid4())[:8]
        self.pending[action_id] = {"action": action, "details": details, "approved": None}
        return action_id
    
    def approve(self, action_id: str) -> bool:
        if action_id in self.pending:
            self.pending[action_id]["approved"] = True
            return True
        return False
    
    def reject(self, action_id: str) -> bool:
        if action_id in self.pending:
            self.pending[action_id]["approved"] = False
            return True
        return False


pending_actions = PendingActions()


@tool
def request_approval(action: str, description: str, impact: str) -> str:
    """
    Request human approval before taking an irreversible action.
    Always use this before: deleting data, sending emails, making purchases,
    deploying code, or any high-impact action.
    Args:
        action: Short name of the action (e.g., "delete_user", "send_campaign")
        description: What exactly will happen
        impact: Potential consequences/impact
    """
    action_id = pending_actions.add(action, {
        "description": description,
        "impact": impact,
    })
    
    console.print(f"\n[bold red]🔴 APPROVAL REQUIRED (ID: {action_id})[/bold red]")
    console.print(f"  Action: {action}")
    console.print(f"  Description: {description}")
    console.print(f"  Impact: {impact}")
    
    # In production: send notification to Slack, email, dashboard, etc.
    # Here we'll auto-approve for demo
    approval = input(f"  Approve action '{action}'? (y/n): ").strip().lower()
    
    if approval == "y":
        pending_actions.approve(action_id)
        return f"Action '{action}' APPROVED (ID: {action_id}). Proceeding."
    else:
        pending_actions.reject(action_id)
        return f"Action '{action}' REJECTED (ID: {action_id}). Do not proceed."


# ══════════════════════════════════════════════════════════════
# H) FULL AGENT WITH ALL TOOL TYPES
# ══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_advanced_tool_agent():
    """Build an agent with all the advanced tool types."""
    
    # Static tools
    static_tools = [
        send_email,
        query_database,
        summarize_and_translate,
        research_and_cite,
    ]
    
    # Stateful tool group
    cart = ShoppingCart()
    cart_tools = cart.get_all_tools()
    
    # Dynamic API tools from config
    api_configs = [
        {
            "name": "get_weather_api",
            "description": "Get current weather for a city",
            "base_url": "https://wttr.in",
            "endpoint": "/{city}?format=3",
            "api_key": None,
        },
        {
            "name": "get_random_joke",
            "description": "Get a random programming joke",
            "base_url": "https://official-joke-api.appspot.com",
            "endpoint": "/jokes/programming/random",
        }
    ]
    dynamic_tools = generate_tools_from_config(api_configs)
    
    all_tools = static_tools + cart_tools + dynamic_tools
    
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    llm_with_tools = llm.bind_tools(all_tools)
    tool_node = ToolNode(all_tools)
    
    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke([
            SystemMessage(content="""You are a helpful AI assistant with many tools.
Available tools:
- send_email: Send emails to users
- query_database: Query user/analytics data
- summarize_and_translate: Summarize + translate text
- research_and_cite: Research topics with citations
- add_to_cart, view_cart, apply_discount: Shopping cart management
- get_weather_api: Current weather
- get_random_joke: Programming jokes

Use the most appropriate tool for each task."""),
        ] + state["messages"])
        return {"messages": [response]}
    
    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"
    
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    
    return graph.compile()


# ══════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════

def run_advanced_tools_demo():
    console.print("[bold magenta]═══ 10: Advanced Tool Patterns ═══[/bold magenta]")
    
    # Demo 1: Structured tools
    console.print("\n[bold cyan]=== Structured Tool Test ===[/bold cyan]")
    result = query_database.invoke({
        "query": "SELECT * FROM users WHERE plan = 'pro'",
        "database": "main",
        "limit": 10,
    })
    console.print(f"DB Result: {result}")
    
    # Demo 2: Stateful tools
    console.print("\n[bold cyan]=== Stateful Cart Tools ===[/bold cyan]")
    cart = ShoppingCart()
    cart_tools = cart.get_all_tools()
    add, view, discount = cart_tools
    
    print(add.invoke({"product_name": "Python Book", "quantity": 2, "price": 49.99}))
    print(add.invoke({"product_name": "Mechanical Keyboard", "quantity": 1, "price": 129.99}))
    print(discount.invoke({"code": "SAVE10"}))
    print(view.invoke({}))
    
    # Demo 3: Dynamic tool generation
    console.print("\n[bold cyan]=== Dynamic Tools ===[/bold cyan]")
    api_configs = [
        {
            "name": "get_cat_facts",
            "description": "Get random cat facts from the API",
            "base_url": "https://catfact.ninja",
            "endpoint": "/fact",
        }
    ]
    dynamic = generate_tools_from_config(api_configs)
    console.print(f"Generated {len(dynamic)} dynamic tool(s): {[t.name for t in dynamic]}")
    
    # Demo 4: Chained tool
    console.print("\n[bold cyan]=== Tool Chaining ===[/bold cyan]")
    result = summarize_and_translate.invoke({
        "text": """LangGraph is a powerful library for building stateful multi-agent applications 
        using language models. It represents agent workflows as graphs where nodes are Python functions 
        and edges control the flow of data. Unlike traditional linear chains, LangGraph supports 
        cycles, branching, and parallel execution.""",
        "target_language": "Spanish",
    })
    console.print(result)
    
    # Demo 5: Full agent
    console.print("\n[bold cyan]=== Full Advanced Tool Agent ===[/bold cyan]")
    agent = build_advanced_tool_agent()
    
    result = agent.invoke({
        "messages": [HumanMessage(content=
            "Query the analytics database and tell me the total revenue across all dates. "
            "Also add a 'LangGraph Book' at $59.99 to my cart and apply the VIP30 discount."
        )]
    })
    
    console.print(f"\n[green]Agent:[/green] {result['messages'][-1].content}")


if __name__ == "__main__":
    run_advanced_tools_demo()
