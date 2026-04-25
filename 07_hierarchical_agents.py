# ============================================================
# 07_hierarchical_agents.py
# ─────────────────────────────────────────────────────────────
# HIERARCHICAL MULTI-AGENT SYSTEM
# Manager agents delegate to worker agents who may delegate further.
# This mirrors how organizations work (CEO → VP → Manager → Worker).
#
# Architecture (3 levels):
#
# Level 1 (Strategic):    CEO_AGENT
#                          /       \
# Level 2 (Tactical):  TECH_DEPT  BIZ_DEPT
#                        / \          / \
# Level 3 (Operational): Dev QA    Sales Mkt  ← leaf workers
#
# Each level can:
# - Do work directly (leaf agents)
# - Delegate to subordinates
# - Aggregate sub-results and report up
# ============================================================

from config import OPENAI_API_KEY, GPT4O_MINI, console

from typing import Annotated, Dict, List, Optional, Any, Tuple
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import json


# ── 1. Core Message Types ─────────────────────────────────────

class Task(BaseModel):
    """A task assigned by a manager to a worker."""
    task_id: str
    description: str
    assigned_to: str
    priority: int = 1
    context: str = ""


class TaskResult(BaseModel):
    """A result returned by a worker to its manager."""
    task_id: str
    agent: str
    result: str
    success: bool
    sub_results: List["TaskResult"] = Field(default_factory=list)


TaskResult.model_rebuild()


# ── 2. Generic Agent Base Class ───────────────────────────────

class HierarchicalAgent:
    """
    Base class for hierarchical agents.
    Can both receive tasks and delegate to subordinates.
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        subordinates: Dict[str, "HierarchicalAgent"] = None,
        tools: list = None,
        temperature: float = 0.2,
    ):
        self.name = name
        self.role = role
        self.subordinates = subordinates or {}
        self.tools = tools or []
        self.llm = ChatOpenAI(
            model=GPT4O_MINI, 
            temperature=temperature, 
            api_key=OPENAI_API_KEY
        )
        self.work_log: List[str] = []
    
    def log(self, message: str):
        indent = "  " * self._get_level()
        console.print(f"{indent}[bold]🔷 {self.name.upper()}[/bold] {message}")
        self.work_log.append(message)
    
    def _get_level(self) -> int:
        """Rough level indicator based on name length."""
        return 0  # Override in subclasses
    
    def should_delegate(self, task: str) -> Optional[str]:
        """
        Decide if this task should be delegated.
        Returns subordinate name or None (do it myself).
        """
        if not self.subordinates:
            return None
        
        sub_names = list(self.subordinates.keys())
        response = self.llm.invoke([
            SystemMessage(content=f"""You are {self.name} ({self.role}).
Your subordinates are: {', '.join(sub_names)}.
Decide if you should delegate this task and to whom.
Reply with exactly: DELEGATE:<subordinate_name> or SELF"""),
            HumanMessage(content=f"Task: {task}"),
        ])
        
        text = response.content.strip()
        if text.startswith("DELEGATE:"):
            delegatee = text.split(":", 1)[1].strip()
            if delegatee in self.subordinates:
                return delegatee
        return None
    
    def do_work(self, task: str, context: str = "") -> str:
        """Execute the task directly (leaf node behavior)."""
        messages = [
            SystemMessage(content=f"""You are {self.name}, a {self.role}.
Do excellent work on the assigned task.
Be specific, thorough, and professional.
Context about the broader goal: {context}"""),
            HumanMessage(content=task),
        ]
        
        if self.tools:
            llm_with_tools = self.llm.bind_tools(self.tools)
            from langgraph.prebuilt import ToolNode
            tool_node = ToolNode(self.tools)
            
            for _ in range(4):
                response = llm_with_tools.invoke(messages)
                messages.append(response)
                if not (hasattr(response, "tool_calls") and response.tool_calls):
                    break
                results = tool_node.invoke({"messages": messages})
                messages.extend(results["messages"])
            return messages[-1].content
        else:
            response = self.llm.invoke(messages)
            return response.content
    
    def execute(self, task: str, context: str = "") -> TaskResult:
        """
        Main entry point: decide to delegate or do directly.
        Aggregates sub-results if delegating.
        """
        self.log(f"Received task: {task[:80]}...")
        
        delegatee_name = self.should_delegate(task)
        
        if delegatee_name and delegatee_name in self.subordinates:
            # Delegate to subordinate
            self.log(f"Delegating to {delegatee_name}...")
            subordinate = self.subordinates[delegatee_name]
            sub_result = subordinate.execute(task, context or task)
            
            # Aggregate/synthesize the sub-result
            synthesis = self.llm.invoke([
                SystemMessage(content=f"You are {self.name}. Synthesize your subordinate's work into your report."),
                HumanMessage(content=f"Task: {task}\n\nSubordinate {delegatee_name}'s work:\n{sub_result.result}"),
            ])
            
            return TaskResult(
                task_id=f"{self.name}-delegated",
                agent=self.name,
                result=synthesis.content,
                success=True,
                sub_results=[sub_result],
            )
        else:
            # Do the work directly
            result = self.do_work(task, context)
            self.log(f"Completed: {result[:60]}...")
            return TaskResult(
                task_id=f"{self.name}-direct",
                agent=self.name,
                result=result,
                success=True,
            )


# ── 3. Specialized Agent Types ────────────────────────────────

@tool
def write_code(specification: str) -> str:
    """Write code based on a specification."""
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    response = llm.invoke([
        SystemMessage(content="Write clean, documented code. Include examples."),
        HumanMessage(content=specification),
    ])
    return response.content


@tool
def run_tests(code: str) -> str:
    """Run tests on provided code."""
    import io, sys
    # Try to run it
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        exec(code, {"__builtins__": __builtins__})
        output = buf.getvalue()
        return f"Tests passed. Output:\n{output}" if output else "Code executed without errors."
    except Exception as e:
        return f"Test failed: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout


@tool
def research_topic(topic: str) -> str:
    """Research a topic and return findings."""
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0.1, api_key=OPENAI_API_KEY)
    response = llm.invoke([
        SystemMessage(content="Provide accurate, detailed research on the topic."),
        HumanMessage(content=f"Research this topic thoroughly: {topic}"),
    ])
    return response.content


@tool
def create_marketing_copy(product: str, audience: str) -> str:
    """Create marketing copy for a product."""
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0.8, api_key=OPENAI_API_KEY)
    response = llm.invoke([
        HumanMessage(content=f"Write compelling marketing copy for: {product}\nTarget audience: {audience}"),
    ])
    return response.content


# ── 4. Build the Hierarchy ────────────────────────────────────

def build_company_hierarchy():
    """
    Build a 3-level company hierarchy:
    
    CEO
    ├── CTO (Tech VP)
    │   ├── Backend Dev
    │   ├── Frontend Dev
    │   └── QA Engineer
    └── CMO (Marketing VP)
        ├── Content Writer
        └── Data Analyst
    """
    
    # ── Level 3: Leaf Workers (no subordinates) ───────────────
    backend_dev = HierarchicalAgent(
        name="backend_dev",
        role="Senior Backend Engineer",
        tools=[write_code, run_tests],
        temperature=0.0,
    )
    
    frontend_dev = HierarchicalAgent(
        name="frontend_dev",
        role="Senior Frontend Engineer",
        tools=[write_code, run_tests],
        temperature=0.0,
    )
    
    qa_engineer = HierarchicalAgent(
        name="qa_engineer",
        role="QA Engineer",
        tools=[run_tests],
        temperature=0.1,
    )
    
    content_writer = HierarchicalAgent(
        name="content_writer",
        role="Content Writer and Copywriter",
        tools=[create_marketing_copy],
        temperature=0.7,
    )
    
    data_analyst = HierarchicalAgent(
        name="data_analyst",
        role="Data Analyst",
        tools=[research_topic],
        temperature=0.1,
    )
    
    # ── Level 2: Department Heads ─────────────────────────────
    cto = HierarchicalAgent(
        name="cto",
        role="Chief Technology Officer — oversees all technical work",
        subordinates={
            "backend_dev":  backend_dev,
            "frontend_dev": frontend_dev,
            "qa_engineer":  qa_engineer,
        },
        temperature=0.2,
    )
    
    cmo = HierarchicalAgent(
        name="cmo",
        role="Chief Marketing Officer — oversees all marketing and content",
        subordinates={
            "content_writer": content_writer,
            "data_analyst":   data_analyst,
        },
        temperature=0.3,
    )
    
    # ── Level 1: CEO ──────────────────────────────────────────
    ceo = HierarchicalAgent(
        name="ceo",
        role="Chief Executive Officer — sets strategy, coordinates all departments",
        subordinates={
            "cto": cto,
            "cmo": cmo,
        },
        temperature=0.3,
    )
    
    return ceo


# ── 5. LangGraph-based Hierarchical System ────────────────────

class HierarchyState(TypedDict):
    task: str
    ceo_plan: str
    tech_output: str
    marketing_output: str
    backend_code: str
    frontend_code: str
    qa_report: str
    content: str
    analytics: str
    messages: Annotated[list, add_messages]
    final_report: str


def build_hierarchy_graph():
    """LangGraph version — explicit nodes for each agent in the hierarchy."""
    
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    
    def make_node(agent_name: str, agent_role: str, input_key: str, output_key: str):
        """Factory for simple agent nodes."""
        def node_fn(state: HierarchyState) -> dict:
            # Get input (from state or task)
            task = state.get(input_key) or state["task"]
            
            console.print(f"  [bold]{agent_name.upper()}[/bold] working...")
            
            response = llm.invoke([
                SystemMessage(content=f"You are {agent_name}, a {agent_role}. Be specific and thorough."),
                HumanMessage(content=task),
            ])
            
            return {output_key: response.content}
        
        node_fn.__name__ = agent_name
        return node_fn
    
    # CEO node: creates the overall plan and sub-tasks
    def ceo_node(state: HierarchyState) -> dict:
        console.print(f"\n[bold red]👑 CEO: Planning task...[/bold red]")
        
        response = llm.invoke([
            SystemMessage(content="""You are the CEO. Break the task into department assignments.
Output a JSON object with keys: tech_task, marketing_task
Each should be a specific, actionable task for that department."""),
            HumanMessage(content=f"Company task: {state['task']}"),
        ])
        
        try:
            # Parse the CEO's task breakdown
            text = response.content
            # Extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                tasks = json.loads(json_match.group())
            else:
                tasks = {
                    "tech_task": f"Build the technical solution for: {state['task']}",
                    "marketing_task": f"Create marketing strategy for: {state['task']}",
                }
        except Exception:
            tasks = {
                "tech_task": f"Build the technical solution for: {state['task']}",
                "marketing_task": f"Create marketing strategy for: {state['task']}",
            }
        
        return {
            "ceo_plan": json.dumps(tasks, indent=2),
            "tech_output": tasks.get("tech_task", ""),
            "marketing_output": tasks.get("marketing_task", ""),
        }
    
    # CTO node: coordinates tech work
    def cto_node(state: HierarchyState) -> dict:
        console.print(f"  [bold blue]🔧 CTO: Coordinating tech team...[/bold blue]")
        
        tech_task = state.get("tech_output") or state["task"]
        response = llm.invoke([
            SystemMessage(content="""You are the CTO. Break the tech task into:
backend (API/database/logic) and frontend (UI/UX) sub-tasks.
Output JSON: {"backend_task": "...", "frontend_task": "...", "qa_task": "..."}"""),
            HumanMessage(content=tech_task),
        ])
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            tasks = json.loads(json_match.group()) if json_match else {}
        except Exception:
            tasks = {}
        
        return {
            "backend_code": tasks.get("backend_task", tech_task),
            "frontend_code": tasks.get("frontend_task", tech_task),
            "qa_report": tasks.get("qa_task", tech_task),
        }
    
    # Leaf worker nodes
    backend_node = make_node(
        "backend_dev", "Python/API developer", "backend_code", "backend_code"
    )
    frontend_node = make_node(
        "frontend_dev", "React/UI developer", "frontend_code", "frontend_code"
    )
    qa_node = make_node(
        "qa_engineer", "QA/testing engineer", "qa_report", "qa_report"
    )
    content_node = make_node(
        "content_writer", "marketing copywriter", "marketing_output", "content"
    )
    analytics_node = make_node(
        "data_analyst", "market research analyst", "marketing_output", "analytics"
    )
    
    # Final synthesis
    def ceo_final_node(state: HierarchyState) -> dict:
        console.print(f"\n[bold red]👑 CEO: Creating final report...[/bold red]")
        
        response = llm.invoke([
            SystemMessage(content="You are the CEO. Synthesize all department outputs into an executive report."),
            HumanMessage(content=f"""Task: {state['task']}

Technical outputs:
Backend: {state.get('backend_code', '')[:300]}
Frontend: {state.get('frontend_code', '')[:300]}
QA: {state.get('qa_report', '')[:300]}

Marketing outputs:
Content: {state.get('content', '')[:300]}
Analytics: {state.get('analytics', '')[:300]}

Write an executive summary integrating all of the above."""),
        ])
        
        return {"final_report": response.content}
    
    # ── Build Graph ───────────────────────────────────────────
    graph = StateGraph(HierarchyState)
    
    graph.add_node("ceo_plan",    ceo_node)
    graph.add_node("cto",         cto_node)
    graph.add_node("backend_dev", backend_node)
    graph.add_node("frontend_dev",frontend_node)
    graph.add_node("qa_engineer", qa_node)
    graph.add_node("content_writer", content_node)
    graph.add_node("data_analyst",   analytics_node)
    graph.add_node("ceo_report",  ceo_final_node)
    
    # Flow: CEO plans → CTO and CMO in parallel
    graph.add_edge(START, "ceo_plan")
    graph.add_edge("ceo_plan", "cto")
    graph.add_edge("ceo_plan", "content_writer")  # CMO path
    graph.add_edge("ceo_plan", "data_analyst")    # CMO path
    
    # CTO → tech workers (parallel)
    graph.add_edge("cto", "backend_dev")
    graph.add_edge("cto", "frontend_dev")
    graph.add_edge("cto", "qa_engineer")
    
    # All workers → CEO final report
    graph.add_edge("backend_dev",     "ceo_report")
    graph.add_edge("frontend_dev",    "ceo_report")
    graph.add_edge("qa_engineer",     "ceo_report")
    graph.add_edge("content_writer",  "ceo_report")
    graph.add_edge("data_analyst",    "ceo_report")
    
    graph.add_edge("ceo_report", END)
    
    return graph.compile()


# ── 6. Main Demo ──────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold magenta]═══ 07: Hierarchical Agent System ═══[/bold magenta]")
    
    task = "Build and launch an AI-powered personal finance app that tracks spending, categorizes transactions, and gives personalized saving advice."
    
    console.print(f"\n[bold white]TASK:[/bold white] {task}\n")
    
    # Option A: Class-based hierarchy (elegant, recursive delegation)
    console.print("\n[bold cyan]=== Approach A: Class-Based Hierarchy ===[/bold cyan]")
    ceo = build_company_hierarchy()
    result = ceo.execute(task)
    
    console.print(f"\n[bold green]CEO Final Report:[/bold green]")
    console.print(result.result[:600])
    
    if result.sub_results:
        console.print(f"\n[dim]Sub-results from {len(result.sub_results)} departments[/dim]")
    
    # Option B: LangGraph-based hierarchy (explicit, visualizable)
    console.print("\n\n[bold cyan]=== Approach B: LangGraph Hierarchy Graph ===[/bold cyan]")
    
    graph = build_hierarchy_graph()
    
    graph_result = graph.invoke({
        "task": task,
        "ceo_plan": "",
        "tech_output": "",
        "marketing_output": "",
        "backend_code": "",
        "frontend_code": "",
        "qa_report": "",
        "content": "",
        "analytics": "",
        "messages": [],
        "final_report": "",
    })
    
    console.print(f"\n[bold green]Final Executive Report:[/bold green]")
    console.print(graph_result["final_report"])
