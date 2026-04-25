# ============================================================
# 04_plan_execute_agent.py
# ─────────────────────────────────────────────────────────────
# PLAN-AND-EXECUTE PATTERN
# Two-stage agent: first make a full plan, then execute each step.
# Unlike ReAct (which plans one step at a time), this agent
# creates an ENTIRE plan upfront, then executes step-by-step.
#
# Architecture:
#
#   User Input
#       ↓
#   [PLANNER] → generates full step-by-step plan
#       ↓
#   [EXECUTOR] → runs each step (with tools)
#       ↓
#   [REPLANNER] → adjusts plan based on results
#       ↓
#   [FINAL] → synthesizes all step results
# ============================================================

from config import OPENAI_API_KEY, GPT4O_MINI, console, print_step

from typing import Annotated, List, Optional, Union
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
import json


# ── 1. State Schema ───────────────────────────────────────────

class Step(BaseModel):
    """A single step in the execution plan."""
    step_number: int
    description: str
    tool_needed: Optional[str] = None  # hint for which tool to use
    depends_on: List[int] = Field(default_factory=list)  # step dependencies


class Plan(BaseModel):
    """The full execution plan."""
    goal: str
    steps: List[Step]
    estimated_steps: int


class StepResult(BaseModel):
    """Result from executing one step."""
    step_number: int
    description: str
    result: str
    success: bool


class PlanExecuteState(TypedDict):
    input: str                              # user's original request
    plan: Optional[Plan]                    # the generated plan
    current_step: int                       # which step we're on
    step_results: List[StepResult]         # completed steps
    messages: Annotated[list, add_messages] # message history
    final_answer: str                       # synthesized output
    should_replan: bool                     # flag to trigger replanning


# ── 2. Tools ──────────────────────────────────────────────────

@tool
def execute_code(code: str) -> str:
    """Run Python code and return output."""
    import io, sys, math
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        exec(code, {"__builtins__": __builtins__, "math": math, "json": json})
        return buffer.getvalue() or "(no output)"
    except Exception as e:
        return f"Error: {e}"
    finally:
        sys.stdout = old_stdout


@tool
def web_search(query: str) -> str:
    """Search the internet for information."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return "\n".join(f"• {r['title']}: {r['body'][:200]}" for r in results)
    except Exception as e:
        return f"Search unavailable: {e}. Mock: '{query}' returns relevant results about the topic."


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    import math
    try:
        result = eval(expression, {"__builtins__": {}}, 
                      {k: getattr(math, k) for k in dir(math) if not k.startswith("_")})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def summarize_text(text: str, max_words: int = 100) -> str:
    """Summarize a long piece of text into key points."""
    words = text.split()
    if len(words) <= max_words:
        return text
    # Simple extractive summary — in production use an LLM
    sentences = text.split(". ")
    key_sentences = sentences[:3]  # take first 3 sentences
    return ". ".join(key_sentences) + "..."


@tool
def create_report(title: str, sections: str) -> str:
    """
    Create a formatted markdown report.
    Args:
        title: Report title
        sections: JSON string with section names and content:
                  [{"name": "Introduction", "content": "..."}]
    """
    try:
        section_list = json.loads(sections)
    except Exception:
        section_list = [{"name": "Content", "content": sections}]
    
    report = f"# {title}\n\n"
    for section in section_list:
        report += f"## {section['name']}\n{section['content']}\n\n"
    return report


ALL_TOOLS = [execute_code, web_search, calculate, summarize_text, create_report]


# ── 3. Planner Node ───────────────────────────────────────────

def make_planner():
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(Plan)
    
    system = """You are a strategic planning agent. Given a user's goal, 
create a detailed step-by-step execution plan.

Break complex goals into concrete, actionable steps.
Each step should be specific and executable by an AI with tools.

Available tools for execution:
- execute_code: Run Python code
- web_search: Search the internet  
- calculate: Do math calculations
- summarize_text: Condense long text
- create_report: Generate formatted reports

Keep steps sequential and clear. 3-7 steps is usually ideal."""

    def planner_node(state: PlanExecuteState) -> dict:
        console.print("\n[bold magenta]📋 PLANNER: Creating execution plan...[/bold magenta]")
        
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Create a plan to accomplish this goal: {state['input']}")
        ]
        
        plan = structured_llm.invoke(messages)
        
        # Display the plan
        console.print(f"\n[bold cyan]Plan: {plan.goal}[/bold cyan]")
        for step in plan.steps:
            console.print(f"  Step {step.step_number}: {step.description}")
            if step.tool_needed:
                console.print(f"    [dim]→ Tool hint: {step.tool_needed}[/dim]")
        
        return {
            "plan": plan,
            "current_step": 1,
            "step_results": [],
            "should_replan": False,
        }
    
    return planner_node


# ── 4. Executor Node ──────────────────────────────────────────

def make_executor():
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)
    
    system = """You are an execution agent. You receive a specific step to execute.
Use the appropriate tools to complete the step.
After completing the step, provide a clear summary of what was accomplished.
Be specific and include actual results/data in your summary."""

    def executor_node(state: PlanExecuteState) -> dict:
        plan = state["plan"]
        current_step_num = state["current_step"]
        
        if current_step_num > len(plan.steps):
            return {}
        
        step = plan.steps[current_step_num - 1]
        
        console.print(f"\n[bold yellow]⚡ EXECUTOR: Running Step {step.step_number}[/bold yellow]")
        console.print(f"  [dim]{step.description}[/dim]")
        
        # Build context from previous steps
        prev_context = ""
        if state["step_results"]:
            prev_context = "\n\nPrevious step results:\n"
            for r in state["step_results"]:
                prev_context += f"- Step {r.step_number}: {r.result[:200]}\n"
        
        prompt = f"""Overall goal: {state['input']}

Current step {step.step_number}: {step.description}
{f'Tool hint: Use {step.tool_needed}' if step.tool_needed else ''}
{prev_context}

Execute this step completely. Use tools as needed."""

        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        
        # Mini agent loop for this step
        for _ in range(4):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            if not (hasattr(response, "tool_calls") and response.tool_calls):
                break
            
            tool_results = tool_node.invoke({"messages": messages})
            messages.extend(tool_results["messages"])
        
        result_text = messages[-1].content
        
        step_result = StepResult(
            step_number=step.step_number,
            description=step.description,
            result=result_text,
            success=True,
        )
        
        console.print(f"  [green]✓ Step {step.step_number} complete[/green]")
        
        return {
            "step_results": state["step_results"] + [step_result],
            "current_step": current_step_num + 1,
            "messages": [AIMessage(content=f"Step {step.step_number} result: {result_text}")],
        }
    
    return executor_node


# ── 5. Replanner Node ─────────────────────────────────────────

class ReplanDecision(BaseModel):
    """Decision: continue or replan."""
    action: str = Field(description="Either 'continue', 'replan', or 'finish'")
    reason: str
    new_steps: Optional[List[str]] = Field(
        default=None,
        description="New steps if replanning (list of step descriptions)"
    )


def make_replanner():
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(ReplanDecision)
    
    system = """You monitor execution progress and decide whether to:
- continue: proceed to next planned step (most common)
- replan: generate new/adjusted steps based on what was learned
- finish: all steps complete, synthesize final answer

Replan only if results revealed the original plan is insufficient or wrong.
Finish when all planned steps are complete."""

    def replanner_node(state: PlanExecuteState) -> dict:
        plan = state["plan"]
        current_step = state["current_step"]
        results = state["step_results"]
        
        results_summary = "\n".join(
            f"Step {r.step_number}: {r.result[:150]}" for r in results
        )
        
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"""Goal: {state['input']}
Steps planned: {len(plan.steps)}
Steps completed: {len(results)}
Current step number: {current_step}

Completed step results:
{results_summary}

Should we continue, replan, or finish?""")
        ]
        
        decision = structured_llm.invoke(messages)
        console.print(f"\n[bold red]🧠 REPLANNER:[/bold red] {decision.action} — {decision.reason}")
        
        if decision.action == "replan" and decision.new_steps:
            # Build new steps from replanner's suggestions
            new_step_objects = [
                Step(step_number=i+1, description=desc)
                for i, desc in enumerate(decision.new_steps)
            ]
            new_plan = Plan(
                goal=plan.goal,
                steps=new_step_objects,
                estimated_steps=len(new_step_objects)
            )
            return {"plan": new_plan, "current_step": 1, "should_replan": True}
        
        return {"should_replan": False}
    
    return replanner_node


# ── 6. Final Synthesis Node ───────────────────────────────────

def make_synthesizer():
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0.3, api_key=OPENAI_API_KEY)
    
    def synthesize_node(state: PlanExecuteState) -> dict:
        console.print("\n[bold green]✅ SYNTHESIZER: Creating final answer...[/bold green]")
        
        results_text = "\n\n".join(
            f"**Step {r.step_number} — {r.description}:**\n{r.result}"
            for r in state["step_results"]
        )
        
        messages = [
            SystemMessage(content="""You are a synthesis expert. Combine the results from all 
executed steps into a comprehensive, well-organized final answer. 
Be clear, concise, and structure the output logically."""),
            HumanMessage(content=f"""Original request: {state['input']}

Results from all steps:
{results_text}

Synthesize these into a complete final response.""")
        ]
        
        response = llm.invoke(messages)
        return {"final_answer": response.content}
    
    return synthesize_node


# ── 7. Router ─────────────────────────────────────────────────

def should_continue_or_finish(state: PlanExecuteState) -> str:
    """After replanner: continue executing, or go to synthesis?"""
    plan = state.get("plan")
    current = state.get("current_step", 1)
    
    if not plan:
        return "synthesize"
    
    if current > len(plan.steps):
        return "synthesize"
    
    return "execute"


# ── 8. Build the Full Graph ───────────────────────────────────

def build_plan_execute_graph():
    planner  = make_planner()
    executor = make_executor()
    replanner = make_replanner()
    synthesizer = make_synthesizer()
    
    graph = StateGraph(PlanExecuteState)
    
    graph.add_node("planner",    planner)
    graph.add_node("execute",    executor)
    graph.add_node("replan",     replanner)
    graph.add_node("synthesize", synthesizer)
    
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "execute")
    graph.add_edge("execute", "replan")
    
    graph.add_conditional_edges(
        "replan",
        should_continue_or_finish,
        {"execute": "execute", "synthesize": "synthesize"}
    )
    
    graph.add_edge("synthesize", END)
    
    return graph.compile()


# ── 9. Demo ───────────────────────────────────────────────────

def run_plan_execute(task: str):
    console.print(f"\n[bold magenta]═══ 04: Plan & Execute Agent ═══[/bold magenta]")
    console.print(f"\n[bold white]TASK:[/bold white] {task}\n")
    
    graph = build_plan_execute_graph()
    
    initial = {
        "input": task,
        "plan": None,
        "current_step": 1,
        "step_results": [],
        "messages": [],
        "final_answer": "",
        "should_replan": False,
    }
    
    result = graph.invoke(initial)
    
    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print("[bold green]FINAL ANSWER:[/bold green]")
    console.print(result["final_answer"])
    console.print(f"[bold green]{'='*60}[/bold green]")
    
    return result


if __name__ == "__main__":
    run_plan_execute(
        "Create a comprehensive analysis of the number 42: "
        "compute its prime factorization, find interesting mathematical properties, "
        "explain its cultural significance, and write a summary report."
    )
    
    run_plan_execute(
        "Build a Python script that generates 20 random numbers between 1-100, "
        "calculates their mean and standard deviation, "
        "finds the min and max, and formats a statistical report."
    )
