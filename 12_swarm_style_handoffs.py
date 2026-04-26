# ============================================================
# 12_swarm_style_handoffs.py
# ─────────────────────────────────────────────────────────────
# SWARM-STYLE AGENT HANDOFFS
# Inspired by OpenAI's Swarm library.
# Agents can TRANSFER control to other agents mid-conversation.
# The transferred-to agent takes over and can transfer again.
#
# Key concept: agents are identified by their SYSTEM PROMPT.
# Handoff = switch the active system prompt + agent.
#
# Example flow:
#   User → Triage Agent → (handoff) → Billing Agent
#                       ← (handoff) → Technical Agent
#
# Real use cases: customer support, sales routing, medical triage
# ============================================================

from config import OPENAI_API_KEY, GPT4O_MINI, console

from typing import Annotated, Dict, List, Optional, Callable
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
import json


# ── 1. Agent Definition ───────────────────────────────────────

class AgentDefinition:
    """Defines an agent: its name, role, tools, and handoff targets."""
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list = None,
        handoff_targets: List[str] = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.handoff_targets = handoff_targets or []


# ── 2. Handoff Tool Factory ───────────────────────────────────

def create_handoff_tool(target_agent_name: str, description: str):
    """
    Create a handoff tool that transfers control to another agent.
    When called, it signals a context switch.
    """
    @tool
    def handoff_tool(reason: str) -> str:
        f"""
        Transfer the conversation to {target_agent_name}.
        {description}
        Args:
            reason: Why you're transferring and what the customer needs
        """
        return json.dumps({
            "handoff": True,
            "target_agent": target_agent_name,
            "reason": reason,
        })
    
    handoff_tool.name = f"transfer_to_{target_agent_name.lower().replace(' ', '_')}"
    handoff_tool.__doc__ = f"Transfer to {target_agent_name}: {description}"
    
    return handoff_tool


# ── 3. Swarm State ────────────────────────────────────────────

class SwarmState(TypedDict):
    messages: Annotated[list, add_messages]
    active_agent: str          # which agent is currently active
    conversation_data: dict    # shared data across agents
    handoff_count: int


# ── 4. Customer Support Swarm Example ────────────────────────

def build_customer_support_swarm():
    """
    Build a customer support system with specialized agents:
    - Triage: routes to the right department
    - Billing: handles payments, subscriptions, refunds
    - Technical: troubleshooting and bug fixes
    - Sales: upgrades, new products
    - Escalation: complex issues for senior support
    """
    
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0.3, api_key=OPENAI_API_KEY)
    
    # ── Specialized Tools ─────────────────────────────────────
    
    @tool
    def lookup_account(customer_email: str) -> str:
        """Look up a customer account by email."""
        mock_accounts = {
            "alice@example.com": {"name": "Alice", "plan": "Pro", "status": "active", "since": "2022-01"},
            "bob@example.com":   {"name": "Bob",   "plan": "Basic", "status": "active", "since": "2023-06"},
            "carol@example.com": {"name": "Carol", "plan": "Enterprise", "status": "past_due", "since": "2021-03"},
        }
        account = mock_accounts.get(customer_email.lower())
        if account:
            return json.dumps(account)
        return json.dumps({"error": "Account not found"})
    
    @tool
    def process_refund(customer_email: str, amount: float, reason: str) -> str:
        """Process a refund for a customer."""
        if amount > 500:
            return json.dumps({"error": "Refunds over $500 require escalation"})
        return json.dumps({
            "refund_id": f"REF-{hash(customer_email) % 10000:04d}",
            "amount": amount,
            "status": "processed",
            "eta": "3-5 business days",
        })
    
    @tool
    def update_subscription(customer_email: str, new_plan: str) -> str:
        """Update a customer's subscription plan."""
        valid_plans = ["Basic", "Pro", "Enterprise"]
        if new_plan not in valid_plans:
            return f"Invalid plan. Choose from: {', '.join(valid_plans)}"
        return json.dumps({"customer": customer_email, "new_plan": new_plan, "effective": "immediately"})
    
    @tool
    def check_system_status(service: str = "all") -> str:
        """Check the operational status of services."""
        statuses = {
            "api":     {"status": "operational", "uptime": "99.99%"},
            "web":     {"status": "operational", "uptime": "99.98%"},
            "database":{"status": "degraded", "uptime": "99.1%", "note": "Elevated latency"},
            "email":   {"status": "operational", "uptime": "100%"},
        }
        if service == "all":
            return json.dumps(statuses)
        return json.dumps(statuses.get(service, {"status": "unknown"}))
    
    @tool
    def run_diagnostic(issue_type: str, customer_email: str) -> str:
        """Run a technical diagnostic for a customer's reported issue."""
        return json.dumps({
            "issue": issue_type,
            "customer": customer_email,
            "diagnostic_result": "Found potential configuration issue",
            "recommended_steps": [
                "Clear browser cache and cookies",
                "Disable browser extensions",
                "Try incognito mode",
                "Check API key permissions",
            ],
            "escalate_if": "Issue persists after following recommended steps",
        })
    
    @tool
    def create_ticket(customer_email: str, priority: str, category: str, description: str) -> str:
        """Create a support ticket for tracking."""
        ticket_id = f"TKT-{abs(hash(customer_email + description)) % 100000:05d}"
        return json.dumps({
            "ticket_id": ticket_id,
            "priority": priority,
            "category": category,
            "status": "open",
            "assigned_to": "Support Team",
            "eta": "24 hours",
        })
    
    @tool
    def get_upgrade_pricing(current_plan: str) -> str:
        """Get upgrade options and pricing for a customer."""
        upgrades = {
            "Basic": {
                "Pro": {"price": "$79/month", "savings": "Save $20 vs monthly", "features": ["Unlimited API calls", "Priority support", "Advanced analytics"]},
                "Enterprise": {"price": "$299/month", "features": ["Everything in Pro", "Dedicated account manager", "SLA guarantee", "Custom integrations"]},
            },
            "Pro": {
                "Enterprise": {"price": "$299/month", "current_discount": "20% off first 3 months", "features": ["SLA", "Dedicated manager", "Custom integrations"]},
            },
        }
        return json.dumps(upgrades.get(current_plan, {"message": "Already on highest plan"}))
    
    # ── Handoff Tools ─────────────────────────────────────────
    transfer_to_billing = create_handoff_tool(
        "billing_agent",
        "Use when customer has payment, subscription, or billing questions"
    )
    
    transfer_to_technical = create_handoff_tool(
        "technical_agent",
        "Use when customer has technical issues, bugs, or API problems"
    )
    
    transfer_to_sales = create_handoff_tool(
        "sales_agent",
        "Use when customer wants to upgrade, add features, or buy more"
    )
    
    transfer_to_escalation = create_handoff_tool(
        "escalation_agent",
        "Use for complex issues, frustrated customers, or matters needing senior review"
    )
    
    transfer_to_triage = create_handoff_tool(
        "triage_agent",
        "Return to triage when the current issue is resolved"
    )
    
    # ── Agent Registry ────────────────────────────────────────
    agents = {
        "triage_agent": AgentDefinition(
            name="triage_agent",
            system_prompt="""You are a friendly Triage Agent for Acme SaaS.
Your job is to:
1. Greet the customer warmly
2. Understand their issue
3. Route them to the right specialist

Routing guide:
- Billing/payment/refund → transfer_to_billing_agent
- Technical/bug/API → transfer_to_technical_agent
- Upgrade/pricing/features → transfer_to_sales_agent
- Complaints/complex/frustrated → transfer_to_escalation_agent

Always be helpful and empathetic. Gather basic info before transferring.""",
            tools=[lookup_account, transfer_to_billing, transfer_to_technical, transfer_to_sales, transfer_to_escalation],
            handoff_targets=["billing_agent", "technical_agent", "sales_agent", "escalation_agent"],
        ),
        
        "billing_agent": AgentDefinition(
            name="billing_agent",
            system_prompt="""You are an expert Billing Specialist for Acme SaaS.
You handle: refunds, subscription changes, payment issues, invoices.

Guidelines:
- Look up the account before taking any action
- Process refunds up to $500 (escalate larger amounts)
- Update subscriptions when requested
- Create tickets for issues you can't resolve immediately
- Transfer back to triage when billing is resolved, or escalate if needed

Always confirm actions before executing them.""",
            tools=[lookup_account, process_refund, update_subscription, create_ticket, transfer_to_escalation, transfer_to_triage],
            handoff_targets=["escalation_agent", "triage_agent"],
        ),
        
        "technical_agent": AgentDefinition(
            name="technical_agent",
            system_prompt="""You are a Technical Support Engineer for Acme SaaS.
You handle: API issues, bugs, integration problems, performance.

Approach:
1. Check system status first
2. Look up the customer account
3. Run diagnostics for the specific issue
4. Provide step-by-step troubleshooting
5. Create a ticket if unresolved
6. Escalate if the issue is a product bug

Be patient, technical, and thorough.""",
            tools=[check_system_status, lookup_account, run_diagnostic, create_ticket, transfer_to_escalation, transfer_to_triage],
            handoff_targets=["escalation_agent", "triage_agent"],
        ),
        
        "sales_agent": AgentDefinition(
            name="sales_agent",
            system_prompt="""You are a friendly Sales Specialist for Acme SaaS.
You handle: upgrades, plan comparisons, custom pricing, new features.

Sales approach:
- Understand the customer's needs and use case
- Show relevant upgrade benefits (not a generic pitch)
- Present pricing clearly
- Handle objections with empathy
- Offer trial extensions when appropriate
- Update subscription when customer is ready

Never be pushy. Focus on value.""",
            tools=[lookup_account, get_upgrade_pricing, update_subscription, transfer_to_billing, transfer_to_triage],
            handoff_targets=["billing_agent", "triage_agent"],
        ),
        
        "escalation_agent": AgentDefinition(
            name="escalation_agent",
            system_prompt="""You are a Senior Support Manager for Acme SaaS.
You handle: complex issues, complaints, escalations, VIP customers.

Your authority:
- Issue refunds of any size
- Offer service credits and goodwill gestures
- Make exceptions to standard policies
- Directly contact engineering for critical bugs
- Provide SLA guarantees

Approach: empathetic, decisive, solution-focused. Make customers feel valued.""",
            tools=[lookup_account, process_refund, update_subscription, create_ticket, transfer_to_triage],
            handoff_targets=["triage_agent"],
        ),
    }
    
    # ── Build the Swarm Graph ─────────────────────────────────
    
    def make_agent_node(agent_def: AgentDefinition):
        """Create a node function for an agent."""
        
        all_agent_tools = agent_def.tools
        llm_with_tools = llm.bind_tools(all_agent_tools)
        
        from langgraph.prebuilt import ToolNode
        
        # Custom tool node that detects handoffs
        def run_tools_with_handoff_detection(state: SwarmState) -> dict:
            last_msg = state["messages"][-1]
            
            if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
                return {}
            
            tool_results = []
            next_agent = None
            handoff_reason = None
            
            for tc in last_msg.tool_calls:
                # Find the matching tool
                matching_tool = next((t for t in all_agent_tools if t.name == tc["name"]), None)
                
                if matching_tool:
                    try:
                        result = matching_tool.invoke(tc["args"])
                        
                        # Check if this is a handoff
                        try:
                            parsed = json.loads(result)
                            if isinstance(parsed, dict) and parsed.get("handoff"):
                                next_agent = parsed["target_agent"]
                                handoff_reason = parsed.get("reason", "")
                        except (json.JSONDecodeError, TypeError):
                            pass
                        
                        tool_results.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        ))
                    except Exception as e:
                        tool_results.append(ToolMessage(
                            content=f"Tool error: {e}",
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        ))
            
            updates: dict = {"messages": tool_results}
            
            if next_agent:
                console.print(f"\n  [bold yellow]🔀 HANDOFF:[/bold yellow] {agent_def.name} → {next_agent}")
                console.print(f"  [dim]Reason: {handoff_reason}[/dim]")
                updates["active_agent"] = next_agent
            
            return updates
        
        def agent_node(state: SwarmState) -> dict:
            active = state.get("active_agent", "triage_agent")
            
            # Only run if we're the active agent
            if active != agent_def.name:
                return {}
            
            console.print(f"\n[bold]🤖 {agent_def.name.upper().replace('_', ' ')}[/bold]")
            
            response = llm_with_tools.invoke(
                [SystemMessage(content=agent_def.system_prompt)] + state["messages"]
            )
            return {"messages": [response]}
        
        return agent_node, run_tools_with_handoff_detection
    
    # Build graph
    graph = StateGraph(SwarmState)
    
    # Add all agent nodes
    for agent_name, agent_def in agents.items():
        agent_node, tool_handler = make_agent_node(agent_def)
        graph.add_node(agent_name, agent_node)
        graph.add_node(f"{agent_name}_tools", tool_handler)
    
    # Routing function: which agent should run?
    def route_to_active_agent(state: SwarmState) -> str:
        active = state.get("active_agent", "triage_agent")
        last_msg = state["messages"][-1] if state["messages"] else None
        
        # If last message has tool calls, run the tools node for the active agent
        if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return f"{active}_tools"
        
        # Otherwise run the active agent
        return active
    
    # Router after tools: did we handoff? If so go to new agent, else stay
    def route_after_tools(agent_name: str):
        def router(state: SwarmState) -> str:
            active = state.get("active_agent", "triage_agent")
            
            # If handoff happened, active_agent changed
            if active != agent_name:
                return active  # go to new agent
            
            # Check if last message has more tool calls
            last_msg = state["messages"][-1] if state["messages"] else None
            if last_msg and isinstance(last_msg, ToolMessage):
                return agent_name  # go back to the agent to process tool results
            
            return END  # conversation turn done
        return router
    
    graph.add_edge(START, "triage_agent")
    
    # Add conditional edges for each agent
    for agent_name, agent_def in agents.items():
        # After agent runs: go to its tools node or END
        graph.add_conditional_edges(
            agent_name,
            lambda state, a=agent_name: f"{a}_tools" if (
                state["messages"] and 
                hasattr(state["messages"][-1], "tool_calls") and 
                state["messages"][-1].tool_calls and
                state.get("active_agent", "triage_agent") == a
            ) else END,
        )
        
        # After tools: route based on handoff or continue
        tool_router = route_after_tools(agent_name)
        graph.add_conditional_edges(
            f"{agent_name}_tools",
            tool_router,
            {**{a: a for a in agents.keys()}, END: END},
        )
    
    return graph.compile()


# ── 5. Simpler Swarm Implementation ───────────────────────────

class SimpleSwarm:
    """
    A simpler, more readable swarm implementation without LangGraph.
    Good for understanding the core concept.
    """
    
    def __init__(self, agents: Dict[str, AgentDefinition], starting_agent: str):
        self.agents = agents
        self.active_agent_name = starting_agent
        self.llm = ChatOpenAI(model=GPT4O_MINI, temperature=0.3, api_key=OPENAI_API_KEY)
        self.conversation_history = []
        self.turn_count = 0
    
    @property
    def active_agent(self) -> AgentDefinition:
        return self.agents[self.active_agent_name]
    
    def chat(self, user_message: str) -> str:
        """Process one turn of conversation."""
        self.turn_count += 1
        self.conversation_history.append(HumanMessage(content=user_message))
        
        console.print(f"\n[white]USER:[/white] {user_message}")
        
        # Run agent with its tools
        llm_with_tools = self.llm.bind_tools(self.active_agent.tools)
        
        max_iterations = 6
        for i in range(max_iterations):
            response = llm_with_tools.invoke(
                [SystemMessage(content=self.active_agent.system_prompt)] + self.conversation_history
            )
            self.conversation_history.append(response)
            
            # No tool calls → agent is done
            if not (hasattr(response, "tool_calls") and response.tool_calls):
                console.print(f"\n[bold green]{self.active_agent_name.upper()}:[/bold green] {response.content}")
                return response.content
            
            # Execute tool calls
            for tc in response.tool_calls:
                matching = next((t for t in self.active_agent.tools if t.name == tc["name"]), None)
                
                if matching:
                    try:
                        result = matching.invoke(tc["args"])
                        
                        # Check for handoff
                        try:
                            parsed = json.loads(str(result))
                            if isinstance(parsed, dict) and parsed.get("handoff"):
                                new_agent = parsed["target_agent"]
                                console.print(f"\n  [yellow]🔀 Handoff: {self.active_agent_name} → {new_agent}[/yellow]")
                                self.active_agent_name = new_agent
                        except (json.JSONDecodeError, TypeError):
                            pass
                        
                        self.conversation_history.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        ))
                    except Exception as e:
                        self.conversation_history.append(ToolMessage(
                            content=f"Error: {e}",
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        ))
        
        return "Agent turn complete."
    
    def run_conversation(self, messages: List[str]) -> List[str]:
        """Run a full multi-turn conversation."""
        responses = []
        for msg in messages:
            response = self.chat(msg)
            responses.append(response)
        return responses


# ── 6. Demo ───────────────────────────────────────────────────

def build_simple_support_swarm() -> SimpleSwarm:
    """Build a simple customer support swarm for demo."""
    
    @tool
    def transfer_to_billing(reason: str) -> str:
        """Transfer customer to the billing department."""
        return json.dumps({"handoff": True, "target_agent": "billing", "reason": reason})
    
    @tool
    def transfer_to_tech(reason: str) -> str:
        """Transfer customer to technical support."""
        return json.dumps({"handoff": True, "target_agent": "technical", "reason": reason})
    
    @tool
    def process_refund_simple(amount: float) -> str:
        """Process a customer refund."""
        return f"Refund of ${amount:.2f} processed successfully. ETA: 3-5 business days."
    
    @tool
    def fix_account(issue: str) -> str:
        """Fix an account technical issue."""
        return f"Fixed account issue: {issue}. Please try again."
    
    @tool
    def transfer_to_triage(reason: str) -> str:
        """Return to triage after resolving the issue."""
        return json.dumps({"handoff": True, "target_agent": "triage", "reason": reason})
    
    agents = {
        "triage": AgentDefinition(
            name="triage",
            system_prompt="You're a helpful support triage agent. Route to billing (payments/refunds) or technical (bugs/errors). Be warm.",
            tools=[transfer_to_billing, transfer_to_tech],
        ),
        "billing": AgentDefinition(
            name="billing",
            system_prompt="You're a billing specialist. Handle refunds and payment issues. Transfer back to triage when done.",
            tools=[process_refund_simple, transfer_to_triage],
        ),
        "technical": AgentDefinition(
            name="technical",
            system_prompt="You're a technical support engineer. Fix account issues and bugs. Transfer back to triage when done.",
            tools=[fix_account, transfer_to_triage],
        ),
    }
    
    return SimpleSwarm(agents, starting_agent="triage")


if __name__ == "__main__":
    console.print("[bold magenta]═══ 12: Swarm-Style Agent Handoffs ═══[/bold magenta]")
    
    swarm = build_simple_support_swarm()
    
    # Simulate a customer support conversation
    conversation = [
        "Hi, I'm having trouble with my account.",
        "I was charged twice for my subscription last month.",
        "Yes, please process a refund of $79.99.",
    ]
    
    console.print("\n[bold cyan]=== Customer Support Conversation ===[/bold cyan]")
    swarm.run_conversation(conversation)
    
    console.print(f"\n[dim]Total turns: {swarm.turn_count} | Final agent: {swarm.active_agent_name}[/dim]")
