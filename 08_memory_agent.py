# ============================================================
# 08_memory_agent.py
# ─────────────────────────────────────────────────────────────
# AGENTS WITH MEMORY — multiple memory types:
#
# 1. Short-term: conversation buffer (recent messages)
# 2. Long-term:  persistent across sessions (vector store)
# 3. Episodic:   remember specific past events
# 4. Semantic:   factual knowledge about entities
# 5. Procedural: how-to knowledge (skills learned over time)
#
# Architecture:
#   User → [Memory Retrieval] → Agent → [Memory Store] → Response
# ============================================================

from config import OPENAI_API_KEY, GPT4O_MINI, console, print_step

from typing import Annotated, Dict, List, Optional, Any
from typing_extensions import TypedDict
from datetime import datetime
import json, uuid

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field


# ── 1. Memory Types ───────────────────────────────────────────

class Memory(BaseModel):
    """A single memory unit."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    memory_type: str   # "episodic", "semantic", "procedural"
    content: str
    importance: float  # 0.0 to 1.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = Field(default_factory=list)
    access_count: int = 0


class MemoryState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    relevant_memories: List[str]
    new_memories: List[str]
    current_intent: str
    entity_knowledge: Dict[str, str]


# ── 2. Memory Store ───────────────────────────────────────────

class MemoryStore:
    """
    Persistent vector-based memory store.
    Uses FAISS for similarity search.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.memories: Dict[str, Memory] = {}  # in-memory store
        self.vector_store: Optional[FAISS] = None
        self._initialize_example_memories()
    
    def _initialize_example_memories(self):
        """Seed with some example memories."""
        example_memories = [
            Memory(
                memory_type="semantic",
                content="The user's name is Alex and they work as a software engineer.",
                importance=0.9,
                tags=["user_info", "name", "job"],
            ),
            Memory(
                memory_type="semantic",
                content="The user prefers Python over JavaScript and likes clean code.",
                importance=0.8,
                tags=["preferences", "programming"],
            ),
            Memory(
                memory_type="episodic",
                content="On 2024-01-15, the user asked for help debugging a FastAPI app that had CORS issues.",
                importance=0.7,
                tags=["fastapi", "debugging", "cors"],
            ),
            Memory(
                memory_type="procedural",
                content="When the user asks about code, provide runnable examples with comments.",
                importance=0.85,
                tags=["communication_style", "code"],
            ),
        ]
        
        for mem in example_memories:
            self.store(mem)
    
    def store(self, memory: Memory) -> str:
        """Store a new memory."""
        self.memories[memory.id] = memory
        
        # Add to vector store for similarity search
        doc = Document(
            page_content=memory.content,
            metadata={
                "id": memory.id,
                "type": memory.memory_type,
                "importance": memory.importance,
                "timestamp": memory.timestamp,
                "tags": ",".join(memory.tags),
            }
        )
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents([doc], self.embeddings)
        else:
            self.vector_store.add_documents([doc])
        
        return memory.id
    
    def recall(self, query: str, k: int = 4, memory_type: str = None) -> List[Memory]:
        """Retrieve memories similar to the query."""
        if not self.vector_store:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k*2)  # get extra, then filter
        except Exception:
            return []
        
        memories = []
        for doc in docs:
            mem_id = doc.metadata.get("id")
            if mem_id and mem_id in self.memories:
                mem = self.memories[mem_id]
                if memory_type is None or mem.memory_type == memory_type:
                    mem.access_count += 1
                    memories.append(mem)
        
        # Sort by importance × recency
        memories.sort(key=lambda m: m.importance, reverse=True)
        return memories[:k]
    
    def forget(self, memory_id: str):
        """Remove a memory (explicit forgetting)."""
        if memory_id in self.memories:
            del self.memories[memory_id]
    
    def consolidate(self):
        """
        Consolidate memories: merge similar ones, forget low-importance old ones.
        This mimics sleep-based memory consolidation in humans.
        """
        # Remove low-importance memories that were never accessed
        to_remove = [
            mid for mid, mem in self.memories.items()
            if mem.importance < 0.3 and mem.access_count == 0
        ]
        for mid in to_remove:
            self.forget(mid)
        
        console.print(f"[dim]Consolidated: removed {len(to_remove)} low-importance memories[/dim]")
    
    def get_summary(self) -> str:
        """Get a summary of stored memories."""
        counts = {}
        for mem in self.memories.values():
            counts[mem.memory_type] = counts.get(mem.memory_type, 0) + 1
        return f"Memory store: {counts} | Total: {len(self.memories)}"


# ── 3. Memory Extraction ──────────────────────────────────────

class MemoryExtraction(BaseModel):
    """Memories to extract from a conversation turn."""
    memories_to_store: List[dict] = Field(
        description="New memories to store. Each dict: {content, type, importance, tags}"
    )
    entity_updates: Dict[str, str] = Field(
        description="Updates to entity knowledge: {entity_name: updated_fact}"
    )


def extract_memories(conversation: str, llm: ChatOpenAI) -> MemoryExtraction:
    """Extract storable memories from a conversation."""
    structured_llm = llm.with_structured_output(MemoryExtraction)
    
    return structured_llm.invoke([
        SystemMessage(content="""Extract memories worth storing from this conversation.

Memory types:
- episodic: specific events, interactions, dates ("User debugged X on DATE")
- semantic: facts, preferences, knowledge ("User prefers Y", "User knows Z")  
- procedural: behavioral patterns, communication preferences ("When user asks X, do Y")

Only store genuinely useful, non-trivial information.
Importance: 0.9 = critical, 0.7 = useful, 0.5 = maybe useful, 0.3 = marginally useful"""),
        HumanMessage(content=f"Extract memories from:\n{conversation}"),
    ])


# ── 4. Build Memory-Augmented Agent ───────────────────────────

def build_memory_agent(user_id: str = "user-default"):
    """Build an agent with full memory capabilities."""
    
    memory_store = MemoryStore(user_id)
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0.3, api_key=OPENAI_API_KEY)
    
    # ── Memory Retrieval Node ─────────────────────────────────
    def memory_retrieval_node(state: MemoryState) -> dict:
        """Fetch relevant memories before the agent responds."""
        # Get the latest user message
        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not user_messages:
            return {"relevant_memories": []}
        
        latest = user_messages[-1].content
        
        # Retrieve relevant memories
        memories = memory_store.recall(latest, k=5)
        
        if memories:
            memory_texts = [
                f"[{m.memory_type.upper()}] {m.content}" 
                for m in memories
            ]
            console.print(f"  [dim]📚 Retrieved {len(memories)} memories[/dim]")
        else:
            memory_texts = []
        
        return {"relevant_memories": memory_texts}
    
    # ── Main Agent Node ───────────────────────────────────────
    def agent_node(state: MemoryState) -> dict:
        """The main agent, enhanced with retrieved memories."""
        memories_text = "\n".join(state.get("relevant_memories", []))
        entity_knowledge = state.get("entity_knowledge", {})
        
        # Build rich system prompt with memory context
        memory_section = ""
        if memories_text:
            memory_section = f"""
RELEVANT MEMORIES (retrieved from long-term memory):
{memories_text}

Use these memories to personalize your response and maintain continuity."""
        
        entity_section = ""
        if entity_knowledge:
            entity_section = f"""
ENTITY KNOWLEDGE:
{json.dumps(entity_knowledge, indent=2)}"""
        
        system = f"""You are a highly personalized AI assistant with persistent memory.
You remember past conversations and use that context to provide better, more relevant help.

{memory_section}
{entity_section}

Guidelines:
- Reference relevant past context naturally (don't just list memories)
- Maintain consistency with what you know about the user
- Update your understanding as you learn new things
- Be helpful, specific, and personalized"""
        
        response = llm.invoke([SystemMessage(content=system)] + state["messages"])
        return {"messages": [response]}
    
    # ── Memory Storage Node ───────────────────────────────────
    def memory_storage_node(state: MemoryState) -> dict:
        """Extract and store new memories after the agent responds."""
        messages = state["messages"]
        
        # Get the last exchange (user + assistant)
        recent = messages[-4:]  # last 2 turns
        conversation_text = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in recent
        ])
        
        try:
            extraction = extract_memories(conversation_text, llm)
            
            stored = []
            for mem_data in extraction.memories_to_store:
                mem = Memory(
                    memory_type=mem_data.get("type", "semantic"),
                    content=mem_data.get("content", ""),
                    importance=float(mem_data.get("importance", 0.5)),
                    tags=mem_data.get("tags", []),
                )
                if mem.importance >= 0.4 and len(mem.content) > 20:
                    memory_store.store(mem)
                    stored.append(mem.content[:60])
            
            if stored:
                console.print(f"  [dim]💾 Stored {len(stored)} new memories[/dim]")
            
            # Update entity knowledge
            new_entities = {**state.get("entity_knowledge", {}), **extraction.entity_updates}
            
            return {
                "new_memories": stored,
                "entity_knowledge": new_entities,
            }
        except Exception as e:
            console.print(f"  [dim]Memory extraction skipped: {e}[/dim]")
            return {"new_memories": []}
    
    # ── Build Graph ───────────────────────────────────────────
    graph = StateGraph(MemoryState)
    
    graph.add_node("retrieve_memory", memory_retrieval_node)
    graph.add_node("agent",           agent_node)
    graph.add_node("store_memory",    memory_storage_node)
    
    graph.add_edge(START,             "retrieve_memory")
    graph.add_edge("retrieve_memory", "agent")
    graph.add_edge("agent",           "store_memory")
    graph.add_edge("store_memory",    END)
    
    # Add checkpointing for conversation continuity
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    
    return compiled, memory_store


# ── 5. Entity Memory (knowledge graph-like) ───────────────────

class EntityMemoryAgent:
    """
    Tracks specific entities (people, projects, concepts) across conversations.
    Builds a knowledge graph of entity relationships.
    """
    
    def __init__(self):
        self.entities: Dict[str, Dict] = {}  # entity → {facts, relationships}
        self.llm = ChatOpenAI(model=GPT4O_MINI, temperature=0.1, api_key=OPENAI_API_KEY)
    
    def update_entity(self, entity: str, fact: str):
        """Add a fact about an entity."""
        if entity not in self.entities:
            self.entities[entity] = {"facts": [], "relationships": {}}
        self.entities[entity]["facts"].append(fact)
    
    def add_relationship(self, entity1: str, relationship: str, entity2: str):
        """Record a relationship between entities."""
        if entity1 not in self.entities:
            self.entities[entity1] = {"facts": [], "relationships": {}}
        self.entities[entity1]["relationships"][relationship] = entity2
    
    def get_entity_context(self, text: str) -> str:
        """Find entities mentioned in text and return their known facts."""
        relevant = []
        for entity, data in self.entities.items():
            if entity.lower() in text.lower():
                facts = "; ".join(data["facts"][-3:])  # last 3 facts
                relevant.append(f"{entity}: {facts}")
        return "\n".join(relevant) if relevant else ""
    
    def chat(self, user_input: str) -> str:
        """Chat with entity memory active."""
        entity_context = self.get_entity_context(user_input)
        
        context_section = f"\nKnown entities:\n{entity_context}" if entity_context else ""
        
        response = self.llm.invoke([
            SystemMessage(content=f"You are a helpful assistant with memory of entities.{context_section}"),
            HumanMessage(content=user_input),
        ])
        
        # Extract and store new entity facts
        extraction = self.llm.invoke([
            SystemMessage(content="""Extract entities and facts from this exchange.
Output JSON: {"entities": [{"name": "X", "fact": "Y"}], "relationships": [{"from": "A", "rel": "B", "to": "C"}]}
If nothing notable: {"entities": [], "relationships": []}"""),
            HumanMessage(content=f"User: {user_input}\nAssistant: {response.content}"),
        ])
        
        try:
            import re
            json_match = re.search(r'\{.*\}', extraction.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for entity_data in data.get("entities", []):
                    self.update_entity(entity_data["name"], entity_data["fact"])
                for rel in data.get("relationships", []):
                    self.add_relationship(rel["from"], rel["rel"], rel["to"])
        except Exception:
            pass
        
        return response.content


# ── 6. Demo ───────────────────────────────────────────────────

def run_memory_demo():
    console.print("[bold magenta]═══ 08: Memory-Augmented Agents ═══[/bold magenta]")
    
    # ── Demo A: Vector Memory Agent ───────────────────────────
    console.print("\n[bold cyan]=== A: Vector Long-Term Memory ===[/bold cyan]")
    
    graph, memory_store = build_memory_agent("user-alex")
    config = {"configurable": {"thread_id": "alex-session-1"}}
    
    def chat(message: str):
        initial = {
            "messages": [HumanMessage(content=message)],
            "user_id": "user-alex",
            "relevant_memories": [],
            "new_memories": [],
            "current_intent": "",
            "entity_knowledge": {},
        }
        result = graph.invoke(initial, config=config)
        response = result["messages"][-1].content
        console.print(f"\n[white]User:[/white] {message}")
        console.print(f"[green]Agent:[/green] {response}\n")
        return response
    
    # These messages will build up memories
    chat("Hi! I'm working on a new Python REST API project using FastAPI.")
    chat("I keep getting a 422 Unprocessable Entity error when I post JSON data.")
    chat("My name is Alex by the way, and I love writing clean, well-documented code.")
    
    # New session — agent should remember from vector memory
    config2 = {"configurable": {"thread_id": "alex-session-2"}}
    
    # Override initial state to simulate new session
    result2 = graph.invoke({
        "messages": [HumanMessage(content="Hey! Can you help me with my FastAPI project again?")],
        "user_id": "user-alex",
        "relevant_memories": [],
        "new_memories": [],
        "current_intent": "",
        "entity_knowledge": {},
    }, config=config2)
    
    console.print(f"\n[white]User (new session):[/white] Hey! Can you help me with my FastAPI project again?")
    console.print(f"[green]Agent (remembers from memory!):[/green] {result2['messages'][-1].content}")
    
    # Memory summary
    console.print(f"\n[dim]{memory_store.get_summary()}[/dim]")
    
    # ── Demo B: Entity Memory ─────────────────────────────────
    console.print("\n\n[bold cyan]=== B: Entity Memory Agent ===[/bold cyan]")
    
    agent = EntityMemoryAgent()
    
    turns = [
        "I'm working on a project called Phoenix with my colleague Sarah.",
        "Sarah is a machine learning engineer and she's building the model layer.",
        "Phoenix is a customer churn prediction system for our company Acme Corp.",
        "Tell me what you know about the Phoenix project and Sarah's role.",
    ]
    
    for turn in turns:
        console.print(f"\n[white]User:[/white] {turn}")
        response = agent.chat(turn)
        console.print(f"[green]Agent:[/green] {response}")
    
    console.print(f"\n[dim]Entity store: {list(agent.entities.keys())}[/dim]")


if __name__ == "__main__":
    run_memory_demo()
