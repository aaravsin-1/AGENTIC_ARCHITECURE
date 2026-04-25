# ============================================================
# 05_reflection_agent.py
# ─────────────────────────────────────────────────────────────
# REFLECTION PATTERN — agents that critique and improve themselves.
# The agent generates a response, then a "critic" evaluates it,
# then the agent revises based on feedback. Loops until good enough.
#
# Two variants shown:
#   A) Self-reflection (same LLM critiques itself)
#   B) Two-agent reflection (separate generator + critic)
#
# Architecture:
#   [GENERATOR] → draft
#       ↓
#   [CRITIC] → critique + score
#       ↓
#   Is score high enough? → No → [GENERATOR] (revise)
#                        → Yes → [FINAL]
# ============================================================

from config import OPENAI_API_KEY, GPT4O_MINI, CLAUDE_HAIKU, console, print_step

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ── 1. State ──────────────────────────────────────────────────

class ReflectionState(TypedDict):
    task: str
    draft: str                       # current draft
    critique: str                    # critic's feedback
    score: float                     # 0.0 to 1.0 quality score
    revision_number: int             # how many revisions made
    messages: Annotated[list, add_messages]
    final_output: str


# ── 2. Critique Schema ────────────────────────────────────────

class Critique(BaseModel):
    """Structured critique from the critic agent."""
    score: float = Field(
        ge=0.0, le=1.0,
        description="Quality score from 0.0 (terrible) to 1.0 (perfect)"
    )
    strengths: list[str] = Field(description="What the draft does well")
    weaknesses: list[str] = Field(description="What needs improvement")
    specific_suggestions: list[str] = Field(description="Concrete changes to make")
    is_acceptable: bool = Field(description="True if score >= 0.8 and ready to publish")


# ── 3A. Self-Reflection Pattern ───────────────────────────────

class SelfReflectionAgent:
    """
    A single LLM that generates, then critiques itself, then improves.
    Simpler but may have blind spots.
    """
    
    def __init__(self, max_revisions: int = 3, target_score: float = 0.85):
        self.llm = ChatOpenAI(
            model=GPT4O_MINI, 
            temperature=0.7, 
            api_key=OPENAI_API_KEY
        )
        self.structured_llm = self.llm.with_structured_output(Critique)
        self.max_revisions = max_revisions
        self.target_score = target_score
    
    def generate(self, task: str, critique: str = "", prev_draft: str = "") -> str:
        """Generate or revise a response."""
        if not prev_draft:
            prompt = f"Task: {task}\n\nWrite a high-quality response:"
        else:
            prompt = f"""Task: {task}

Your previous draft:
{prev_draft}

Critique received:
{critique}

Now write an improved version addressing all the feedback:"""
        
        response = self.llm.invoke([
            SystemMessage(content="You are an expert writer and analyst. Produce excellent, accurate, well-structured content."),
            HumanMessage(content=prompt),
        ])
        return response.content
    
    def critique(self, task: str, draft: str) -> Critique:
        """Critique the draft objectively."""
        prompt = f"""Task given: {task}

Draft to evaluate:
{draft}

Evaluate this draft objectively. Be honest and specific."""
        
        return self.structured_llm.invoke([
            SystemMessage(content="""You are a strict quality reviewer. 
Your job is to find flaws and suggest specific improvements.
Do not be lenient — a score of 0.8+ means it's genuinely publication-ready."""),
            HumanMessage(content=prompt),
        ])
    
    def run(self, task: str) -> dict:
        """Run the full reflection loop."""
        console.print(f"\n[bold cyan]Self-Reflection Agent[/bold cyan]")
        console.print(f"Task: {task}\n")
        
        draft = ""
        critique_text = ""
        history = []
        
        for revision in range(self.max_revisions + 1):
            # Generate
            console.print(f"[bold yellow]📝 Generating {'initial draft' if revision == 0 else f'revision {revision}'}...[/bold yellow]")
            draft = self.generate(task, critique_text, draft if revision > 0 else "")
            
            history.append({
                "revision": revision,
                "draft": draft[:200] + "...",
            })
            
            # Critique
            console.print(f"[bold red]🔍 Critiquing...[/bold red]")
            critique = self.critique(task, draft)
            critique_text = "\n".join([
                f"Score: {critique.score:.2f}",
                f"Weaknesses: {', '.join(critique.weaknesses)}",
                f"Suggestions: {', '.join(critique.specific_suggestions)}",
            ])
            
            console.print(f"  Score: {critique.score:.2f} | Acceptable: {critique.is_acceptable}")
            console.print(f"  Strengths: {', '.join(critique.strengths[:2])}")
            console.print(f"  Weaknesses: {', '.join(critique.weaknesses[:2])}")
            
            if critique.is_acceptable or revision >= self.max_revisions:
                console.print(f"\n[bold green]✅ Done after {revision} revision(s). Score: {critique.score:.2f}[/bold green]")
                break
        
        return {"task": task, "final_draft": draft, "final_score": critique.score, "history": history}


# ── 3B. Two-Agent Reflection (Generator + Critic) ─────────────

def build_two_agent_reflection_graph(max_revisions: int = 3, target_score: float = 0.82):
    """
    Separate generator and critic agents in a LangGraph loop.
    The critic can be a different model (e.g., Claude critiques GPT output).
    """
    
    # Different models for generator and critic (interesting!)
    generator_llm = ChatOpenAI(model=GPT4O_MINI, temperature=0.7, api_key=OPENAI_API_KEY)
    critic_llm = ChatOpenAI(model=GPT4O_MINI, temperature=0.1, api_key=OPENAI_API_KEY)
    # For real diversity: critic_llm = ChatAnthropic(model=CLAUDE_HAIKU, api_key=ANTHROPIC_API_KEY)
    
    structured_critic = critic_llm.with_structured_output(Critique)
    
    # ── Generator Node ────────────────────────────────────────
    def generator_node(state: ReflectionState) -> dict:
        revision = state.get("revision_number", 0)
        
        if revision == 0:
            # Initial generation
            prompt = f"""Create a high-quality response to the following task.
Be thorough, accurate, well-structured, and engaging.

Task: {state['task']}"""
        else:
            # Revision based on critique
            prompt = f"""Improve your previous draft based on the critique.
Address every weakness and suggestion specifically.

Task: {state['task']}

Previous draft:
{state['draft']}

Critique and feedback:
{state['critique']}
Score was: {state['score']:.2f}/1.0

Write an improved version that addresses all feedback:"""
        
        console.print(f"\n[bold yellow]📝 GENERATOR: {'Writing initial draft' if revision == 0 else f'Writing revision {revision}'}...[/bold yellow]")
        
        response = generator_llm.invoke([
            SystemMessage(content="""You are an expert content creator. 
Write clear, accurate, well-structured, and engaging responses.
When revising, explicitly address each piece of feedback."""),
            HumanMessage(content=prompt),
        ])
        
        return {
            "draft": response.content,
            "messages": [AIMessage(content=f"[GENERATOR v{revision}]: {response.content[:100]}...")],
        }
    
    # ── Critic Node ───────────────────────────────────────────
    def critic_node(state: ReflectionState) -> dict:
        console.print(f"[bold red]🔍 CRITIC: Evaluating draft...[/bold red]")
        
        critique = structured_critic.invoke([
            SystemMessage(content="""You are a demanding quality critic.
Evaluate the draft against the task requirements.
Be specific, fair, and constructive.
Score 0.8+ only for genuinely excellent work."""),
            HumanMessage(content=f"""Task: {state['task']}

Draft to evaluate:
{state['draft']}

Provide detailed critique:"""),
        ])
        
        critique_text = "\n".join([
            f"SCORE: {critique.score:.2f}",
            "",
            "✅ STRENGTHS:",
            *[f"  • {s}" for s in critique.strengths],
            "",
            "❌ WEAKNESSES:",
            *[f"  • {w}" for w in critique.weaknesses],
            "",
            "💡 SUGGESTIONS:",
            *[f"  • {s}" for s in critique.specific_suggestions],
        ])
        
        console.print(f"  [{'green' if critique.is_acceptable else 'red'}]Score: {critique.score:.2f} | Acceptable: {critique.is_acceptable}[/{'green' if critique.is_acceptable else 'red'}]")
        
        return {
            "critique": critique_text,
            "score": critique.score,
            "revision_number": state.get("revision_number", 0) + 1,
            "messages": [HumanMessage(content=f"[CRITIC]: Score {critique.score:.2f}. {critique_text[:100]}...")],
        }
    
    # ── Finalizer Node ────────────────────────────────────────
    def finalize_node(state: ReflectionState) -> dict:
        console.print(f"\n[bold green]✅ FINAL: Polishing output...[/bold green]")
        
        response = generator_llm.invoke([
            SystemMessage(content="Polish this response for final publication. Fix any remaining issues. Keep what works well."),
            HumanMessage(content=f"Final polish of:\n\n{state['draft']}"),
        ])
        
        return {"final_output": response.content}
    
    # ── Router ────────────────────────────────────────────────
    def should_revise(state: ReflectionState) -> str:
        score = state.get("score", 0.0)
        revision = state.get("revision_number", 0)
        
        if score >= target_score:
            console.print(f"  [green]→ Score {score:.2f} ≥ {target_score} — Finalizing![/green]")
            return "finalize"
        
        if revision >= max_revisions:
            console.print(f"  [yellow]→ Max revisions ({max_revisions}) reached — Finalizing![/yellow]")
            return "finalize"
        
        console.print(f"  [yellow]→ Score {score:.2f} < {target_score} — Revising...[/yellow]")
        return "generate"
    
    # ── Build Graph ───────────────────────────────────────────
    graph = StateGraph(ReflectionState)
    
    graph.add_node("generate",  generator_node)
    graph.add_node("critique",  critic_node)
    graph.add_node("finalize",  finalize_node)
    
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "critique")
    
    graph.add_conditional_edges(
        "critique",
        should_revise,
        {"generate": "generate", "finalize": "finalize"}
    )
    
    graph.add_edge("finalize", END)
    
    return graph.compile()


# ── 4. Constitutional AI Pattern ──────────────────────────────
# The AI critiques its output against a set of "constitutional" principles

class ConstitutionalAgent:
    """
    Inspired by Anthropic's Constitutional AI approach.
    The agent revises its output to comply with a set of principles.
    """
    
    CONSTITUTION = [
        "The response must be factually accurate and not contain hallucinations.",
        "The response must be helpful and directly address the user's question.",
        "The response must be safe and not encourage harmful behavior.",
        "The response must be honest and acknowledge uncertainty where it exists.",
        "The response must be concise — no unnecessary padding or repetition.",
    ]
    
    def __init__(self):
        self.llm = ChatOpenAI(model=GPT4O_MINI, temperature=0.3, api_key=OPENAI_API_KEY)
    
    def generate_initial(self, task: str) -> str:
        response = self.llm.invoke([HumanMessage(content=task)])
        return response.content
    
    def check_principle(self, response: str, principle: str, task: str) -> tuple[bool, str]:
        """Check if response violates a principle and suggest fix."""
        check_prompt = f"""Principle: {principle}

Task: {task}
Response to check: {response}

Does this response violate the principle?
Answer with:
VIOLATES: [yes/no]
REASONING: [brief explanation]
FIX: [how to fix it, or "No fix needed"]"""
        
        result = self.llm.invoke([HumanMessage(content=check_prompt)])
        text = result.content
        
        violates = "VIOLATES: yes" in text.lower()
        fix_line = [l for l in text.split("\n") if l.startswith("FIX:")]
        fix = fix_line[0].replace("FIX:", "").strip() if fix_line else ""
        
        return violates, fix
    
    def revise_against_principle(self, response: str, principle: str, fix: str, task: str) -> str:
        """Revise the response to comply with the principle."""
        revise_prompt = f"""Original task: {task}

Current response: {response}

This response violates the principle: "{principle}"
Suggested fix: {fix}

Rewrite the response to comply with the principle while keeping everything else good:"""
        
        result = self.llm.invoke([HumanMessage(content=revise_prompt)])
        return result.content
    
    def run(self, task: str) -> str:
        """Run constitutional AI loop."""
        console.print(f"\n[bold cyan]Constitutional AI Agent[/bold cyan]")
        console.print(f"Task: {task}\n")
        
        response = self.generate_initial(task)
        console.print(f"[dim]Initial response generated[/dim]")
        
        for i, principle in enumerate(self.CONSTITUTION, 1):
            console.print(f"[bold blue]Checking principle {i}:[/bold blue] {principle[:60]}...")
            
            violates, fix = self.check_principle(response, principle, task)
            
            if violates and fix and fix != "No fix needed":
                console.print(f"  [red]⚠ Violation found. Revising...[/red]")
                response = self.revise_against_principle(response, principle, fix, task)
                console.print(f"  [green]✓ Revised[/green]")
            else:
                console.print(f"  [green]✓ Compliant[/green]")
        
        return response


# ── 5. Main Demo ──────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold magenta]═══ 05: Reflection Agents ═══[/bold magenta]")
    
    task = "Explain quantum entanglement to a 10-year-old, then to a physics PhD student."
    
    # Demo 1: Self-reflection
    agent_a = SelfReflectionAgent(max_revisions=2, target_score=0.80)
    result_a = agent_a.run(task)
    print_step("Self-Reflection Final Output", result_a["final_draft"][:500] + "...")
    
    # Demo 2: Two-agent reflection graph
    console.print("\n\n[bold cyan]Two-Agent Reflection:[/bold cyan]")
    graph = build_two_agent_reflection_graph(max_revisions=2)
    
    initial_state: ReflectionState = {
        "task": task,
        "draft": "",
        "critique": "",
        "score": 0.0,
        "revision_number": 0,
        "messages": [],
        "final_output": "",
    }
    
    result = graph.invoke(initial_state)
    print_step("Two-Agent Reflection Final Output", result["final_output"][:500] + "...")
    
    # Demo 3: Constitutional AI
    console.print("\n\n[bold cyan]Constitutional AI:[/bold cyan]")
    agent_c = ConstitutionalAgent()
    constitutional_result = agent_c.run(
        "What's the fastest way to lose 20 pounds? Give specific advice."
    )
    print_step("Constitutional Output", constitutional_result[:500] + "...")
