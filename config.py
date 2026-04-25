# ============================================================
# config.py — Shared configuration & fake credentials
# Replace fake keys with real ones before running
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys (fake placeholders — swap for real) ──────────────
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY",    "sk-fake-openai-key-1234567890abcdef")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "sk-ant-fake-anthropic-key-abcdef1234")
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY",    "tvly-fake-tavily-key-abcdef1234")
LANGSMITH_API_KEY   = os.getenv("LANGSMITH_API_KEY", "lsv2-fake-langsmith-key-abcdef")

# ── Model names ───────────────────────────────────────────────
GPT4O               = "gpt-4o"
GPT4O_MINI          = "gpt-4o-mini"
CLAUDE_SONNET       = "claude-3-5-sonnet-20241022"
CLAUDE_HAIKU        = "claude-3-5-haiku-20241022"

# ── LangSmith tracing (optional but recommended) ──────────────
os.environ["OPENAI_API_KEY"]            = OPENAI_API_KEY
os.environ["ANTHROPIC_API_KEY"]         = ANTHROPIC_API_KEY
os.environ["TAVILY_API_KEY"]            = TAVILY_API_KEY
os.environ["LANGCHAIN_TRACING_V2"]      = "true"
os.environ["LANGCHAIN_API_KEY"]         = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"]         = "agentic-ai-examples"

# ── Pretty printing helper ────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

def print_step(title: str, content: str, color: str = "cyan"):
    console.print(Panel(content, title=f"[bold {color}]{title}[/bold {color}]", border_style=color))

def print_agent_message(agent: str, message: str):
    colors = {
        "supervisor": "red",
        "researcher": "blue",
        "coder":      "green",
        "critic":     "yellow",
        "planner":    "magenta",
        "executor":   "cyan",
        "user":       "white",
    }
    color = colors.get(agent.lower(), "white")
    console.print(f"\n[bold {color}][{agent.upper()}][/bold {color}] {message}")
