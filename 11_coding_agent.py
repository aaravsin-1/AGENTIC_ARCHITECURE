# ============================================================
# 11_coding_agent.py
# ─────────────────────────────────────────────────────────────
# AUTONOMOUS CODING AGENT
# Writes, executes, tests, and debugs code in a loop.
# Inspired by SWE-agent, Devin, and OpenAI's code interpreter.
#
# Features:
# - Writes code based on specification
# - Runs it and reads output
# - Debugs errors automatically
# - Writes and runs unit tests
# - Refactors for quality
# - Documents the code
#
# Architecture:
#   Spec → Write → Run → (Error?) → Debug → Run
#              ↓                         ↑
#         (Tests pass?) → Document → Output
# ============================================================

from config import OPENAI_API_KEY, GPT4O_MINI, console, print_step

import io, sys, os, ast, textwrap
from typing import Annotated, Dict, List, Optional, Tuple
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ── 1. State ──────────────────────────────────────────────────

class CodingState(TypedDict):
    task: str                    # what to build
    code: str                    # current code
    test_code: str               # unit tests
    execution_output: str        # last run output
    error: str                   # last error (if any)
    debug_attempts: int          # how many times we've debugged
    tests_passed: bool           # did all tests pass?
    documentation: str           # generated docs
    messages: Annotated[list, add_messages]
    phase: str                   # "write", "test", "debug", "document", "done"


# ── 2. Safe Code Execution ────────────────────────────────────

class ExecutionResult(BaseModel):
    stdout: str
    stderr: str
    success: bool
    return_value: Optional[str] = None


def execute_python_code(code: str, timeout: int = 30) -> ExecutionResult:
    """
    Safely execute Python code in an isolated environment.
    Captures stdout/stderr. Returns structured result.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stdout_capture = io.StringIO()
    sys.stderr = stderr_capture = io.StringIO()
    
    result = ExecutionResult(stdout="", stderr="", success=False)
    
    try:
        # Create a clean namespace
        namespace = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
        }
        
        # Import commonly needed modules
        exec("import math, json, re, os, sys, time, datetime, random, collections, itertools", namespace)
        
        exec(code, namespace)
        
        result.stdout = stdout_capture.getvalue()
        result.success = True
        
    except SyntaxError as e:
        result.stderr = f"SyntaxError at line {e.lineno}: {e.msg}\n{e.text}"
        result.success = False
    except Exception as e:
        import traceback
        result.stderr = traceback.format_exc()
        result.success = False
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return result


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """Check if code has valid Python syntax before running."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def extract_code_from_response(response: str) -> str:
    """Extract code blocks from LLM response."""
    import re
    
    # Try to find ```python ... ``` blocks
    python_blocks = re.findall(r'```python\n?(.*?)```', response, re.DOTALL)
    if python_blocks:
        return "\n\n".join(python_blocks).strip()
    
    # Try ``` ... ``` blocks (no language specified)
    code_blocks = re.findall(r'```\n?(.*?)```', response, re.DOTALL)
    if code_blocks:
        return "\n\n".join(code_blocks).strip()
    
    # Return the whole response if no blocks found
    return response.strip()


# ── 3. Agent Nodes ────────────────────────────────────────────

def build_coding_agent():
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    
    # ── Node: Write Initial Code ──────────────────────────────
    def write_code_node(state: CodingState) -> dict:
        console.print(f"\n[bold cyan]📝 WRITER: Generating code...[/bold cyan]")
        
        response = llm.invoke([
            SystemMessage(content="""You are an expert Python developer.
Write clean, working, well-structured Python code.

Requirements:
- Use descriptive variable and function names
- Add docstrings to all functions and classes
- Include type hints
- Handle edge cases
- Add print statements to show results/examples
- Make the code self-contained and runnable

IMPORTANT: Output ONLY the Python code in a ```python code block. No explanations."""),
            HumanMessage(content=f"Write Python code to accomplish this task:\n\n{state['task']}"),
        ])
        
        code = extract_code_from_response(response.content)
        console.print(f"[dim]Generated {len(code.split(chr(10)))} lines of code[/dim]")
        
        return {
            "code": code,
            "phase": "run",
            "messages": [AIMessage(content=f"Code written:\n```python\n{code}\n```")],
        }
    
    # ── Node: Run Code ────────────────────────────────────────
    def run_code_node(state: CodingState) -> dict:
        console.print(f"\n[bold yellow]⚡ RUNNER: Executing code...[/bold yellow]")
        
        code = state["code"]
        
        # Syntax check first
        valid, syntax_error = validate_python_syntax(code)
        if not valid:
            console.print(f"  [red]Syntax error: {syntax_error}[/red]")
            return {
                "error": f"SyntaxError: {syntax_error}",
                "execution_output": "",
                "phase": "debug",
            }
        
        result = execute_python_code(code)
        
        if result.success:
            console.print(f"  [green]✓ Code ran successfully[/green]")
            if result.stdout:
                console.print(f"  Output: {result.stdout[:200]}")
            return {
                "execution_output": result.stdout,
                "error": "",
                "phase": "write_tests",
            }
        else:
            console.print(f"  [red]✗ Execution error:[/red] {result.stderr[:150]}")
            return {
                "execution_output": result.stdout,
                "error": result.stderr,
                "phase": "debug",
            }
    
    # ── Node: Debug Code ──────────────────────────────────────
    def debug_code_node(state: CodingState) -> dict:
        debug_attempt = state.get("debug_attempts", 0) + 1
        console.print(f"\n[bold red]🐛 DEBUGGER: Attempt {debug_attempt}...[/bold red]")
        
        if debug_attempt > 4:
            console.print("  [red]Max debug attempts reached. Giving up.[/red]")
            return {"phase": "document", "debug_attempts": debug_attempt}
        
        response = llm.invoke([
            SystemMessage(content="""You are an expert Python debugger.
Analyze the error, find the root cause, and fix the code.

Important:
- Read the full error traceback carefully
- Identify the exact line causing the error
- Fix ONLY what's broken, keep everything else
- Output ONLY the fixed Python code in ```python blocks"""),
            HumanMessage(content=f"""Task: {state['task']}

Current code:
```python
{state['code']}
```

Error encountered:
{state['error']}

Output before error:
{state.get('execution_output', '(none)')}

Fix the code and output the corrected version:"""),
        ])
        
        fixed_code = extract_code_from_response(response.content)
        
        console.print(f"  [dim]Debugged code has {len(fixed_code.split(chr(10)))} lines[/dim]")
        
        return {
            "code": fixed_code,
            "debug_attempts": debug_attempt,
            "phase": "run",
            "messages": [AIMessage(content=f"Debug attempt {debug_attempt}: Fixed code")],
        }
    
    # ── Node: Write Tests ─────────────────────────────────────
    def write_tests_node(state: CodingState) -> dict:
        console.print(f"\n[bold magenta]🧪 TESTER: Writing unit tests...[/bold magenta]")
        
        response = llm.invoke([
            SystemMessage(content="""Write comprehensive pytest unit tests for the provided code.

Test requirements:
- Test normal/happy path
- Test edge cases (empty input, None, zero, negative numbers, etc.)
- Test error cases (invalid input should raise exceptions)
- Use descriptive test function names (test_xxx_when_yyy_then_zzz)
- Use assertions with helpful error messages
- Import pytest and any needed modules

Output ONLY the test code in ```python blocks. Do not repeat the original code."""),
            HumanMessage(content=f"""Write tests for this code:

```python
{state['code']}
```

Task it was built for: {state['task']}"""),
        ])
        
        test_code = extract_code_from_response(response.content)
        
        # Run the tests
        console.print(f"  [dim]Running tests...[/dim]")
        
        # Combine original code + tests for execution
        full_test_code = f"""
{state['code']}

# ── Unit Tests ──
import sys

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def run_test(self, name, fn):
        try:
            fn()
            self.passed += 1
            print(f"  ✓ {{name}}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append(f"{{name}}: {{e}}")
            print(f"  ✗ {{name}}: {{e}}")
        except Exception as e:
            self.failed += 1
            self.errors.append(f"{{name}}: {{type(e).__name__}}: {{e}}")
            print(f"  ✗ {{name}}: {{type(e).__name__}}: {{e}}")

runner = TestRunner()

# Extract test functions from test code
{test_code.replace("import pytest", "").replace("@pytest.mark.", "# @pytest.mark.")}

# Find and run all test_* functions
test_fns = [name for name, obj in list(locals().items()) if name.startswith('test_') and callable(obj)]
print(f"\\nRunning {{len(test_fns)}} tests:")
for fn_name in test_fns:
    runner.run_test(fn_name, locals()[fn_name])

print(f"\\nResults: {{runner.passed}} passed, {{runner.failed}} failed")
if runner.failed > 0:
    print("Failures:")
    for e in runner.errors:
        print(f"  - {{e}}")
"""
        
        result = execute_python_code(full_test_code)
        
        tests_passed = result.success and "failed" not in result.stdout.lower().replace("0 failed", "")
        
        console.print(f"  {'[green]✓ Tests passed[/green]' if tests_passed else '[red]✗ Some tests failed[/red]'}")
        if result.stdout:
            console.print(f"  {result.stdout[:300]}")
        
        return {
            "test_code": test_code,
            "tests_passed": tests_passed,
            "execution_output": state.get("execution_output", "") + "\n\nTest output:\n" + result.stdout,
            "phase": "document",
        }
    
    # ── Node: Document Code ───────────────────────────────────
    def document_code_node(state: CodingState) -> dict:
        console.print(f"\n[bold green]📚 DOCUMENTER: Creating documentation...[/bold green]")
        
        response = llm.invoke([
            SystemMessage(content="""Generate comprehensive documentation for the provided code.
Include:
1. Overview: what the code does and why
2. Installation/dependencies needed
3. Usage examples with expected output
4. Function/class reference (parameters, returns, exceptions)
5. Design decisions and architecture notes
6. Limitations and known issues
7. Possible improvements

Format as clean Markdown."""),
            HumanMessage(content=f"""Document this code:

Task: {state['task']}

```python
{state['code']}
```

Test results: {'✅ All tests passed' if state.get('tests_passed') else '⚠️ Some tests failed'}
Output when run: {state.get('execution_output', '')[:300]}"""),
        ])
        
        return {
            "documentation": response.content,
            "phase": "done",
            "messages": [AIMessage(content="Documentation complete!")],
        }
    
    # ── Routing ───────────────────────────────────────────────
    def route_after_run(state: CodingState) -> str:
        phase = state.get("phase", "run")
        return phase  # "debug", "write_tests", "document"
    
    def route_after_write(state: CodingState) -> str:
        return state.get("phase", "run")
    
    # ── Build Graph ───────────────────────────────────────────
    graph = StateGraph(CodingState)
    
    graph.add_node("write",       write_code_node)
    graph.add_node("run",         run_code_node)
    graph.add_node("debug",       debug_code_node)
    graph.add_node("write_tests", write_tests_node)
    graph.add_node("document",    document_code_node)
    
    graph.add_edge(START, "write")
    graph.add_edge("write", "run")
    
    graph.add_conditional_edges(
        "run",
        route_after_run,
        {"debug": "debug", "write_tests": "write_tests", "document": "document"}
    )
    
    # After debug: try running again
    graph.add_edge("debug", "run")
    
    # Tests → document
    graph.add_edge("write_tests", "document")
    graph.add_edge("document", END)
    
    return graph.compile()


# ── 4. Interactive Coding Session ─────────────────────────────

class InteractiveCodingSession:
    """
    A REPL-like session where the agent can iteratively improve code
    based on user feedback.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
        self.current_code = ""
        self.history = []
    
    def generate_or_modify(self, instruction: str) -> str:
        """Generate new code or modify existing code."""
        if self.current_code:
            prompt = f"""Current code:
```python
{self.current_code}
```

Modification request: {instruction}

Apply the modification and output the complete updated code in ```python blocks."""
        else:
            prompt = f"Write Python code for: {instruction}\n\nOutput in ```python blocks."
        
        messages = [
            SystemMessage(content="You are an expert Python developer. Write clean, documented, working code."),
            *self.history,
            HumanMessage(content=prompt),
        ]
        
        response = self.llm.invoke(messages)
        self.history.extend([HumanMessage(content=instruction), response])
        
        new_code = extract_code_from_response(response.content)
        self.current_code = new_code
        return new_code
    
    def run_current(self) -> ExecutionResult:
        """Run the current code."""
        return execute_python_code(self.current_code)
    
    def explain(self) -> str:
        """Explain the current code."""
        response = self.llm.invoke([
            HumanMessage(content=f"Explain this code clearly:\n```python\n{self.current_code}\n```"),
        ])
        return response.content
    
    def refactor(self, goal: str = "improve readability and performance") -> str:
        """Refactor the code for a specific goal."""
        return self.generate_or_modify(f"Refactor the code to {goal}. Keep same functionality.")


# ── 5. Demo ───────────────────────────────────────────────────

def run_coding_demo():
    console.print("[bold magenta]═══ 11: Autonomous Coding Agent ═══[/bold magenta]")
    
    graph = build_coding_agent()
    
    # Task 1: Data structures
    console.print("\n[bold white]Task 1: Data Structures[/bold white]")
    result1 = graph.invoke({
        "task": """Implement a MinHeap data structure in Python with:
- insert(value): add element
- extract_min(): remove and return smallest element  
- peek(): view smallest without removing
- heapify(list): build heap from existing list
- Demonstrate with 10 random numbers, show the sorted extraction""",
        "code": "", "test_code": "", "execution_output": "", "error": "",
        "debug_attempts": 0, "tests_passed": False, "documentation": "",
        "messages": [], "phase": "write",
    })
    
    print_step("Final Code", result1["code"][:800] + ("..." if len(result1["code"]) > 800 else ""))
    print_step("Execution Output", result1["execution_output"][:500])
    print_step("Documentation", result1["documentation"][:600])
    
    # Task 2: Algorithm
    console.print("\n[bold white]Task 2: Algorithm[/bold white]")
    result2 = graph.invoke({
        "task": """Implement Dijkstra's shortest path algorithm:
- Graph represented as adjacency list with weights
- Find shortest path from source to all nodes
- Return distances dict and path reconstruction dict
- Show example with a 6-node weighted graph""",
        "code": "", "test_code": "", "execution_output": "", "error": "",
        "debug_attempts": 0, "tests_passed": False, "documentation": "",
        "messages": [], "phase": "write",
    })
    
    print_step("Algorithm Output", result2["execution_output"][:400])
    
    # Task 3: Interactive session
    console.print("\n[bold cyan]=== Interactive Coding Session ===[/bold cyan]")
    session = InteractiveCodingSession()
    
    steps = [
        "Create a simple task manager class that can add, complete, and list tasks",
        "Add priority levels (high, medium, low) and a method to get high-priority incomplete tasks",
        "Add due dates and a method to get overdue tasks",
    ]
    
    for step in steps:
        console.print(f"\n[white]Instruction:[/white] {step}")
        code = session.generate_or_modify(step)
        result = session.run_current()
        
        if result.success:
            console.print(f"[green]✓ Works![/green] {result.stdout[:150] if result.stdout else '(no output)'}")
        else:
            console.print(f"[red]✗ Error:[/red] {result.stderr[:150]}")
    
    console.print(f"\n[bold green]Final code ({len(session.current_code.split(chr(10)))} lines):[/bold green]")
    console.print(session.current_code[:500])


if __name__ == "__main__":
    run_coding_demo()
