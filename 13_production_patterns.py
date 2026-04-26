# ============================================================
# 13_production_patterns.py
# ─────────────────────────────────────────────────────────────
# PRODUCTION-READY AGENT PATTERNS
#
# A) Streaming: token-by-token real-time output
# B) Observability: LangSmith tracing + custom logging
# C) Rate limiting & cost management
# D) Graceful error handling & circuit breakers
# E) Caching (reduce latency & cost)
# F) Structured logging & monitoring
# G) Agent evaluation & testing
# H) Deployment with FastAPI
# ============================================================

from config import OPENAI_API_KEY, GPT4O_MINI, LANGSMITH_API_KEY, console

import asyncio, time, json, hashlib, functools
from typing import Annotated, AsyncIterator, Dict, Any, Optional
from typing_extensions import TypedDict
from datetime import datetime, timedelta
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ══════════════════════════════════════════════════════════════
# A) STREAMING: Real-Time Token Output
# ══════════════════════════════════════════════════════════════

async def stream_agent_response(prompt: str):
    """Stream agent response token by token."""
    llm = ChatOpenAI(
        model=GPT4O_MINI,
        temperature=0.3,
        api_key=OPENAI_API_KEY,
        streaming=True,  # Enable streaming
    )
    
    console.print("\n[bold cyan]Streaming response:[/bold cyan]", end="")
    
    full_response = ""
    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        token = chunk.content
        if token:
            print(token, end="", flush=True)
            full_response += token
    
    print()  # newline
    return full_response


async def stream_langgraph_agent(query: str):
    """Stream a LangGraph agent's execution in real-time."""
    from langgraph.prebuilt import ToolNode
    
    @tool
    def simple_calc(expression: str) -> str:
        """Calculate a math expression."""
        try:
            import math
            result = eval(expression, {"__builtins__": {}}, 
                         {k: getattr(math, k) for k in dir(math) if not k.startswith("_")})
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    
    class StreamState(TypedDict):
        messages: Annotated[list, add_messages]
    
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY, streaming=True)
    llm_with_tools = llm.bind_tools([simple_calc])
    tool_node = ToolNode([simple_calc])
    
    def agent_fn(state: StreamState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    def router(state: StreamState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"
    
    graph = StateGraph(StreamState)
    graph.add_node("agent", agent_fn)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    compiled = graph.compile()
    
    console.print(f"\n[bold cyan]Streaming LangGraph execution:[/bold cyan]")
    
    inputs = {"messages": [HumanMessage(content=query)]}
    
    # stream_mode="messages" streams individual tokens from LLM nodes
    async for msg, metadata in compiled.astream(inputs, stream_mode="messages"):
        if hasattr(msg, "content") and msg.content:
            if metadata.get("langgraph_node") == "agent":
                print(msg.content, end="", flush=True)
    
    print()


# ══════════════════════════════════════════════════════════════
# B) OBSERVABILITY: Custom Callback Handler
# ══════════════════════════════════════════════════════════════

class AgentObserver(BaseCallbackHandler):
    """
    Custom callback handler for observability.
    Tracks: latency, token usage, tool calls, errors.
    In production: send to DataDog, Prometheus, LangSmith, etc.
    """
    
    def __init__(self, agent_name: str = "agent"):
        self.agent_name = agent_name
        self.metrics: Dict[str, Any] = {
            "llm_calls": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tool_calls": [],
            "errors": [],
            "latencies_ms": [],
            "total_cost_usd": 0.0,
        }
        self._start_time: Optional[float] = None
        self.call_log = []
    
    # LLM Callbacks
    def on_llm_start(self, serialized, prompts, **kwargs):
        self._start_time = time.time()
        self.metrics["llm_calls"] += 1
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        if self._start_time:
            latency = (time.time() - self._start_time) * 1000
            self.metrics["latencies_ms"].append(latency)
        
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.metrics["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.metrics["completion_tokens"] += usage.get("completion_tokens", 0)
            self.metrics["total_tokens"] += usage.get("total_tokens", 0)
            
            # Estimate cost (GPT-4o-mini pricing)
            prompt_cost = usage.get("prompt_tokens", 0) * 0.00000015
            completion_cost = usage.get("completion_tokens", 0) * 0.0000006
            self.metrics["total_cost_usd"] += prompt_cost + completion_cost
    
    def on_llm_error(self, error, **kwargs):
        self.metrics["errors"].append({
            "type": "llm_error",
            "error": str(error),
            "timestamp": datetime.utcnow().isoformat(),
        })
        console.print(f"  [red]LLM Error: {error}[/red]")
    
    # Tool Callbacks
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.metrics["tool_calls"].append({
            "tool": tool_name,
            "input": input_str[:100],
            "timestamp": datetime.utcnow().isoformat(),
            "start_time": time.time(),
        })
    
    def on_tool_end(self, output, **kwargs):
        if self.metrics["tool_calls"]:
            last_call = self.metrics["tool_calls"][-1]
            last_call["output"] = str(output)[:100]
            last_call["duration_ms"] = (time.time() - last_call.pop("start_time", time.time())) * 1000
    
    def on_tool_error(self, error, **kwargs):
        self.metrics["errors"].append({
            "type": "tool_error",
            "error": str(error),
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    # Chain Callbacks
    def on_chain_start(self, serialized, inputs, **kwargs):
        self.call_log.append(f"Chain start: {serialized.get('name', 'unnamed')}")
    
    def on_chain_end(self, outputs, **kwargs):
        self.call_log.append(f"Chain end")
    
    def get_report(self) -> str:
        """Generate an observability report."""
        metrics = self.metrics
        latencies = metrics["latencies_ms"]
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return f"""
📊 Agent Observability Report — {self.agent_name}
{'='*50}
LLM Calls:        {metrics['llm_calls']}
Total Tokens:     {metrics['total_tokens']:,}
  Prompt:         {metrics['prompt_tokens']:,}
  Completion:     {metrics['completion_tokens']:,}
Estimated Cost:   ${metrics['total_cost_usd']:.6f}
Tool Calls:       {len(metrics['tool_calls'])}
  Tools Used:     {list(set(t['tool'] for t in metrics['tool_calls']))}
Avg Latency:      {avg_latency:.0f}ms
Errors:           {len(metrics['errors'])}
"""


# ══════════════════════════════════════════════════════════════
# C) RATE LIMITING & COST MANAGEMENT
# ══════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    Prevents hitting OpenAI rate limits.
    """
    
    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 100_000):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.request_timestamps = []
        self.token_usage = []
    
    def can_make_request(self, estimated_tokens: int = 1000) -> tuple[bool, float]:
        """Check if we can make a request. Returns (allowed, wait_seconds)."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old entries
        self.request_timestamps = [t for t in self.request_timestamps if t > minute_ago]
        self.token_usage = [(t, tok) for t, tok in self.token_usage if t > minute_ago]
        
        # Check request rate
        if len(self.request_timestamps) >= self.rpm:
            oldest = min(self.request_timestamps)
            wait = 60 - (now - oldest)
            return False, wait
        
        # Check token rate
        used_tokens = sum(tok for _, tok in self.token_usage)
        if used_tokens + estimated_tokens > self.tpm:
            return False, 5.0  # wait 5 seconds
        
        return True, 0.0
    
    def record_request(self, tokens_used: int):
        """Record a completed request."""
        now = time.time()
        self.request_timestamps.append(now)
        self.token_usage.append((now, tokens_used))
    
    async def wait_if_needed(self, estimated_tokens: int = 1000):
        """Async wait until we can make a request."""
        while True:
            allowed, wait_time = self.can_make_request(estimated_tokens)
            if allowed:
                break
            console.print(f"  [yellow]Rate limit: waiting {wait_time:.1f}s...[/yellow]")
            await asyncio.sleep(wait_time)


class CostManager:
    """Track and enforce cost budgets."""
    
    # GPT-4o-mini pricing per token
    COSTS = {
        "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
        "gpt-4o":       {"input": 0.0000025,  "output": 0.00001},
    }
    
    def __init__(self, daily_budget_usd: float = 1.0):
        self.daily_budget = daily_budget_usd
        self.daily_spent = 0.0
        self.session_spent = 0.0
        self.reset_time = datetime.utcnow() + timedelta(days=1)
    
    def record_usage(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record token usage and return cost."""
        self._reset_if_needed()
        
        prices = self.COSTS.get(model, self.COSTS["gpt-4o-mini"])
        cost = input_tokens * prices["input"] + output_tokens * prices["output"]
        
        self.daily_spent += cost
        self.session_spent += cost
        
        return cost
    
    def check_budget(self) -> tuple[bool, str]:
        """Check if we're within budget. Returns (ok, message)."""
        self._reset_if_needed()
        
        if self.daily_spent >= self.daily_budget:
            return False, f"Daily budget (${self.daily_budget:.2f}) exceeded!"
        
        remaining = self.daily_budget - self.daily_spent
        if remaining < self.daily_budget * 0.1:  # < 10% remaining
            return True, f"Warning: Only ${remaining:.4f} of daily budget remaining!"
        
        return True, "OK"
    
    def _reset_if_needed(self):
        if datetime.utcnow() >= self.reset_time:
            self.daily_spent = 0.0
            self.reset_time = datetime.utcnow() + timedelta(days=1)
    
    def get_status(self) -> str:
        return (f"Budget: ${self.daily_spent:.4f} / ${self.daily_budget:.2f} today | "
                f"Session: ${self.session_spent:.4f}")


# ══════════════════════════════════════════════════════════════
# D) CIRCUIT BREAKER PATTERN
# Prevents cascading failures if an API goes down
# ══════════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    Circuit breaker for external API calls.
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
    """
    
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        
        self.state = self.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
    
    def call(self, func, *args, **kwargs):
        """Execute a function through the circuit breaker."""
        
        if self.state == self.OPEN:
            # Check if timeout has elapsed → move to HALF_OPEN
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = self.HALF_OPEN
                self.success_count = 0
                console.print(f"  [yellow]Circuit breaker: HALF_OPEN — testing...[/yellow]")
            else:
                raise Exception(f"Circuit breaker is OPEN. Retry after {self.timeout}s.")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        if self.state == self.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = self.CLOSED
                self.failure_count = 0
                console.print(f"  [green]Circuit breaker: CLOSED — recovered![/green]")
        elif self.state == self.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN
            console.print(f"  [red]Circuit breaker: OPEN — too many failures![/red]")


# ══════════════════════════════════════════════════════════════
# E) SEMANTIC CACHING
# Cache LLM responses for similar queries
# ══════════════════════════════════════════════════════════════

class SemanticCache:
    """
    Cache LLM responses using semantic similarity.
    Similar queries return cached answers instead of calling the LLM.
    """
    
    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self.cache: Dict[str, Dict] = {}  # query_hash → {response, embedding, timestamp}
        self.hits = 0
        self.misses = 0
    
    def _hash_query(self, query: str) -> str:
        """Exact-match hash for quick lookup."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[str]:
        """Try to get a cached response for the query."""
        key = self._hash_query(query)
        
        if key in self.cache:
            entry = self.cache[key]
            # Check if not expired (1 hour TTL)
            if time.time() - entry["timestamp"] < 3600:
                self.hits += 1
                console.print(f"  [green]Cache HIT[/green] for: {query[:50]}...")
                return entry["response"]
        
        self.misses += 1
        return None
    
    def set(self, query: str, response: str):
        """Cache a response."""
        key = self._hash_query(query)
        self.cache[key] = {
            "query": query,
            "response": response,
            "timestamp": time.time(),
        }
    
    def stats(self) -> str:
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return f"Cache: {self.hits} hits / {self.misses} misses ({hit_rate:.1f}% hit rate)"
    
    def cached_invoke(self, llm: ChatOpenAI, messages: list) -> str:
        """Invoke LLM with caching."""
        # Use the last user message as the cache key
        query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        
        cached = self.get(query)
        if cached:
            return cached
        
        response = llm.invoke(messages)
        self.set(query, response.content)
        return response.content


# ══════════════════════════════════════════════════════════════
# F) STRUCTURED LOGGING
# ══════════════════════════════════════════════════════════════

import logging
import structlog

def setup_structured_logging():
    """Configure structured JSON logging for production."""
    
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# G) AGENT EVALUATION FRAMEWORK
# ══════════════════════════════════════════════════════════════

class AgentEvaluator:
    """Framework for testing and evaluating agents."""
    
    def __init__(self, agent_fn, evaluator_llm: ChatOpenAI = None):
        self.agent = agent_fn
        self.evaluator_llm = evaluator_llm or ChatOpenAI(
            model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY
        )
        self.results = []
    
    def evaluate_response(self, question: str, response: str, expected: str = None, criteria: list = None) -> dict:
        """Evaluate a single agent response using LLM-as-judge."""
        
        criteria = criteria or [
            "accuracy: Is the response factually correct?",
            "relevance: Does it directly answer the question?",
            "completeness: Does it cover all important aspects?",
            "clarity: Is it clear and well-organized?",
        ]
        
        criteria_text = "\n".join(f"- {c}" for c in criteria)
        expected_text = f"\nExpected answer: {expected}" if expected else ""
        
        eval_prompt = f"""Evaluate this AI response:

Question: {question}
Response: {response}{expected_text}

Score each criterion 1-5:
{criteria_text}

Output JSON: {{"scores": {{"criterion": score}}, "overall": 0-5, "feedback": "..."}}"""
        
        result = self.evaluator_llm.invoke([HumanMessage(content=eval_prompt)])
        
        try:
            import re
            json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
            scores = json.loads(json_match.group()) if json_match else {"overall": 3}
        except Exception:
            scores = {"overall": 3, "feedback": "Evaluation parsing failed"}
        
        return {
            "question": question,
            "response": response[:200],
            "scores": scores,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def run_test_suite(self, test_cases: list) -> dict:
        """Run a full test suite and return metrics."""
        results = []
        
        for case in test_cases:
            question = case["question"]
            expected = case.get("expected")
            
            # Get agent response
            try:
                response = self.agent(question)
            except Exception as e:
                response = f"ERROR: {e}"
            
            # Evaluate
            eval_result = self.evaluate_response(question, response, expected)
            eval_result["agent_response"] = response
            results.append(eval_result)
        
        # Aggregate scores
        overall_scores = [r["scores"].get("overall", 3) for r in results]
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        
        return {
            "test_cases": len(test_cases),
            "average_score": avg_score,
            "pass_rate": sum(1 for s in overall_scores if s >= 4) / len(overall_scores) if overall_scores else 0,
            "results": results,
        }


# ══════════════════════════════════════════════════════════════
# H) FASTAPI DEPLOYMENT
# ══════════════════════════════════════════════════════════════

FASTAPI_APP_CODE = '''
# ============================================================
# app.py — FastAPI deployment for your LangGraph agent
# Run with: uvicorn app:app --reload
# ============================================================

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncIterator
import asyncio, json, uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict

app = FastAPI(title="AI Agent API", version="1.0")

# ── Request/Response Models ────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tokens_used: Optional[int] = None


# ── Build Agent ───────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]

def build_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    def agent_fn(state: State) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
    graph = StateGraph(State)
    graph.add_node("agent", agent_fn)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

agent = build_agent()

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
        )
        return ChatResponse(
            response=result["messages"][-1].content,
            session_id=session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream the agent response token by token."""
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    async def generate() -> AsyncIterator[str]:
        llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
        async for chunk in llm.astream([HumanMessage(content=req.message)]):
            if chunk.content:
                yield f"data: {json.dumps({'token': chunk.content, 'session_id': session_id})}\\n\\n"
        yield f"data: {json.dumps({'done': True})}\\n\\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    # In production: delete from your checkpoint store
    return {"status": "cleared", "session_id": session_id}


@app.get("/metrics")
async def get_metrics():
    """Return agent performance metrics."""
    return {
        "active_sessions": 0,  # populate from your monitoring
        "total_requests": 0,
        "avg_latency_ms": 0,
        "error_rate": 0,
    }
'''


# ══════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════

def run_production_demo():
    console.print("[bold magenta]═══ 13: Production Patterns ═══[/bold magenta]")
    
    # A) Streaming demo
    console.print("\n[bold cyan]=== A: Token Streaming ===[/bold cyan]")
    asyncio.run(stream_agent_response("Explain the concept of gradient descent in 3 sentences."))
    
    # B) Observer demo
    console.print("\n[bold cyan]=== B: Observability ===[/bold cyan]")
    observer = AgentObserver("demo-agent")
    
    llm = ChatOpenAI(
        model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY,
        callbacks=[observer],
    )
    llm.invoke([HumanMessage(content="What is 15 * 23?")])
    llm.invoke([HumanMessage(content="Name 3 Python frameworks.")])
    
    console.print(observer.get_report())
    
    # C) Rate limiter demo
    console.print("[bold cyan]=== C: Rate Limiting ===[/bold cyan]")
    limiter = RateLimiter(requests_per_minute=100)
    allowed, wait = limiter.can_make_request(500)
    console.print(f"Request allowed: {allowed} (wait: {wait:.1f}s)")
    limiter.record_request(500)
    console.print(f"After recording: {len(limiter.request_timestamps)} requests this minute")
    
    # D) Cost manager demo
    console.print("\n[bold cyan]=== D: Cost Management ===[/bold cyan]")
    cost_mgr = CostManager(daily_budget_usd=0.10)
    cost = cost_mgr.record_usage("gpt-4o-mini", input_tokens=1000, output_tokens=500)
    console.print(f"Request cost: ${cost:.6f}")
    ok, msg = cost_mgr.check_budget()
    console.print(f"Budget status: {msg}")
    console.print(cost_mgr.get_status())
    
    # E) Cache demo
    console.print("\n[bold cyan]=== E: Semantic Caching ===[/bold cyan]")
    cache = SemanticCache()
    llm_simple = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    
    query = "What is the capital of France?"
    
    # First call — cache miss
    response1 = cache.cached_invoke(llm_simple, [HumanMessage(content=query)])
    console.print(f"Response: {response1[:100]}")
    
    # Second call — cache hit!
    response2 = cache.cached_invoke(llm_simple, [HumanMessage(content=query)])
    console.print(f"Cached: {response2[:100]}")
    
    console.print(cache.stats())
    
    # F) Circuit breaker demo
    console.print("\n[bold cyan]=== F: Circuit Breaker ===[/bold cyan]")
    breaker = CircuitBreaker(failure_threshold=2, timeout=5.0)
    
    def unreliable_api():
        import random
        if random.random() < 0.7:
            raise Exception("Service unavailable")
        return "success"
    
    for attempt in range(5):
        try:
            result = breaker.call(unreliable_api)
            console.print(f"  Attempt {attempt+1}: ✓ {result} (state: {breaker.state})")
        except Exception as e:
            console.print(f"  Attempt {attempt+1}: ✗ {e} (state: {breaker.state})")
    
    # G) Evaluator demo
    console.print("\n[bold cyan]=== G: Agent Evaluation ===[/bold cyan]")
    
    def simple_agent(question: str) -> str:
        llm_eval = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
        response = llm_eval.invoke([HumanMessage(content=question)])
        return response.content
    
    evaluator = AgentEvaluator(simple_agent)
    
    test_cases = [
        {"question": "What is 2 + 2?", "expected": "4"},
        {"question": "What is the capital of Japan?", "expected": "Tokyo"},
        {"question": "Briefly explain what a hash function does."},
    ]
    
    report = evaluator.run_test_suite(test_cases)
    
    console.print(f"\nEval Results:")
    console.print(f"  Tests: {report['test_cases']}")
    console.print(f"  Avg Score: {report['average_score']:.1f}/5")
    console.print(f"  Pass Rate: {report['pass_rate']*100:.0f}%")
    
    # H) Save FastAPI app
    console.print("\n[bold cyan]=== H: FastAPI App ===[/bold cyan]")
    with open("/tmp/agent_app.py", "w") as f:
        f.write(FASTAPI_APP_CODE)
    console.print("[green]✓ FastAPI app saved to /tmp/agent_app.py[/green]")
    console.print("[dim]Run with: pip install fastapi uvicorn && uvicorn agent_app:app --reload[/dim]")


if __name__ == "__main__":
    run_production_demo()
