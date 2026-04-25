python

# ============================================================
# 09_agentic_rag.py
# ─────────────────────────────────────────────────────────────
# AGENTIC RAG (Retrieval-Augmented Generation)
# Unlike naive RAG (retrieve → generate), agentic RAG can:
# - Decide WHEN to retrieve
# - Rephrase queries for better retrieval
# - Retrieve MULTIPLE times iteratively
# - Grade retrieved docs and retry if needed
# - Combine multiple knowledge sources
#
# Patterns:
#   A) Corrective RAG (CRAG): grades docs, web-searches if bad
#   B) Self-RAG: decides whether to retrieve at all
#   C) Adaptive RAG: routes to different retrievers by query type
# ============================================================

from config import OPENAI_API_KEY, GPT4O_MINI, console

from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import json


# ── 1. Build Knowledge Base ───────────────────────────────────

SAMPLE_DOCUMENTS = [
    """
    LangChain is a framework for developing applications powered by language models.
    It provides tools for chaining LLM calls, managing prompts, and integrating with external data.
    LangChain supports multiple LLM providers including OpenAI, Anthropic, and Hugging Face.
    Key components: chains, agents, tools, memory, and document loaders.
    LangChain Expression Language (LCEL) is the new declarative syntax for building chains.
    """,
    """
    LangGraph is a library built on top of LangChain for building stateful, multi-actor applications.
    It models agent workflows as graphs where nodes are functions and edges control flow.
    LangGraph supports cycles (loops), branching, parallel execution, and human-in-the-loop.
    State is maintained across graph traversals using TypedDicts and reducers.
    Key features: checkpointing, streaming, time-travel debugging, and persistence.
    """,
    """
    Vector databases store high-dimensional embeddings for semantic similarity search.
    Popular vector databases: Pinecone, Weaviate, Chroma, Qdrant, FAISS (local).
    FAISS (Facebook AI Similarity Search) is an efficient local vector library.
    Embeddings convert text into numerical vectors that capture semantic meaning.
    OpenAI's text-embedding-3-small model produces 1536-dimensional vectors.
    Cosine similarity measures how similar two vectors are, ranging from -1 to 1.
    """,
    """
    Retrieval-Augmented Generation (RAG) improves LLM accuracy by providing relevant context.
    Standard RAG pipeline: embed query → search vector DB → append docs → generate response.
    Agentic RAG improves on this: the agent decides when/how to retrieve adaptively.
    Corrective RAG (CRAG) grades retrieved documents and searches the web if docs are poor.
    Self-RAG generates retrieval decisions as special tokens to control the pipeline.
    Advanced RAG techniques: HyDE, query decomposition, fusion retrieval, reranking.
    """,
    """
    Prompt engineering is the practice of designing effective prompts for language models.
    Key techniques: chain-of-thought (CoT), few-shot examples, system instructions.
    Advanced techniques: tree of thoughts, least-to-most prompting, self-consistency.
    ReAct (Reason + Act) interleaves reasoning steps with tool calls.
    Constitutional AI uses a set of principles to guide model behavior.
    Prompt injection is an attack where malicious prompts override system instructions.
    """,
    """
    OpenAI's GPT-4 is a multimodal large language model released in March 2023.
    GPT-4o (Omni) is a faster, cheaper version supporting text, image, and audio.
    Anthropic's Claude 3.5 Sonnet is known for strong coding and reasoning abilities.
    Google's Gemini 1.5 Pro supports a 1 million token context window.
    These models use transformer architecture with attention mechanisms.
    Fine-tuning adapts pre-trained models to specific tasks using additional training data.
    """,
]


def build_vector_store() -> FAISS:
    """Build a local FAISS vector store from sample documents."""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    
    docs = []
    for i, text in enumerate(SAMPLE_DOCUMENTS):
        chunks = splitter.split_text(text.strip())
        for j, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": f"doc_{i}", "chunk": j}
            ))
    
    vector_store = FAISS.from_documents(docs, embeddings)
    console.print(f"[dim]Vector store built with {len(docs)} chunks[/dim]")
    return vector_store


# ── 2. Document Grader ────────────────────────────────────────

class DocumentGrade(BaseModel):
    """Grade for a retrieved document."""
    relevant: bool = Field(description="True if doc is relevant to the query")
    score: float = Field(ge=0.0, le=1.0, description="Relevance score")
    reason: str = Field(description="Why this score was given")


def grade_document(doc: Document, query: str, llm: ChatOpenAI) -> DocumentGrade:
    """Grade a retrieved document for relevance."""
    grader = llm.with_structured_output(DocumentGrade)
    
    return grader.invoke([
        SystemMessage(content="""Grade the relevance of a retrieved document to a user query.
Score 0.8+ for highly relevant docs, 0.5-0.8 for somewhat relevant, <0.5 for irrelevant."""),
        HumanMessage(content=f"Query: {query}\n\nDocument:\n{doc.page_content}"),
    ])


# ── 3A. Corrective RAG (CRAG) ─────────────────────────────────

class CRAGState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    graded_docs: List[Document]
    web_search_needed: bool
    web_results: str
    generation: str
    attempts: int


def build_crag_graph(vector_store: FAISS):
    """
    Corrective RAG: grade docs after retrieval, web-search if poor quality.
    """
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # ── Node: Retrieve ────────────────────────────────────────
    def retrieve_node(state: CRAGState) -> dict:
        console.print(f"  [blue]📥 Retrieving docs for:[/blue] {state['question'][:60]}...")
        
        docs = retriever.invoke(state["question"])
        return {"retrieved_docs": docs, "attempts": state.get("attempts", 0) + 1}
    
    # ── Node: Grade Documents ─────────────────────────────────
    def grade_docs_node(state: CRAGState) -> dict:
        console.print(f"  [yellow]🔍 Grading {len(state['retrieved_docs'])} docs...[/yellow]")
        
        good_docs = []
        for doc in state["retrieved_docs"]:
            grade = grade_document(doc, state["question"], llm)
            if grade.relevant:
                good_docs.append(doc)
                console.print(f"    [green]✓[/green] Score {grade.score:.2f}: {doc.page_content[:50]}...")
            else:
                console.print(f"    [red]✗[/red] Score {grade.score:.2f}: {doc.page_content[:50]}...")
        
        # Need web search if fewer than 2 good docs
        web_needed = len(good_docs) < 2
        
        return {
            "graded_docs": good_docs,
            "web_search_needed": web_needed,
        }
    
    # ── Node: Web Search ──────────────────────────────────────
    def web_search_node(state: CRAGState) -> dict:
        console.print(f"  [magenta]🌐 Poor retrieval — searching web...[/magenta]")
        
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(state["question"], max_results=3))
            web_text = "\n\n".join(f"{r['title']}: {r['body']}" for r in results)
        except Exception as e:
            web_text = f"Web search failed: {e}. Using LLM knowledge instead."
        
        return {"web_results": web_text}
    
    # ── Node: Rewrite Query (for better retrieval) ────────────
    def rewrite_query_node(state: CRAGState) -> dict:
        console.print(f"  [cyan]✏️  Rewriting query for better retrieval...[/cyan]")
        
        response = llm.invoke([
            SystemMessage(content="""You are a query optimization expert.
Rewrite the query to be more specific and retrieve better results from a vector database.
Focus on key technical terms and concepts."""),
            HumanMessage(content=f"Original query: {state['question']}\nWrite a better query:"),
        ])
        
        return {"question": response.content.strip()}
    
    # ── Node: Generate Answer ─────────────────────────────────
    def generate_node(state: CRAGState) -> dict:
        console.print(f"  [green]💡 Generating answer...[/green]")
        
        # Combine docs and web results
        context_parts = []
        
        if state.get("graded_docs"):
            doc_texts = "\n\n".join(d.page_content for d in state["graded_docs"])
            context_parts.append(f"From knowledge base:\n{doc_texts}")
        
        if state.get("web_results"):
            context_parts.append(f"From web search:\n{state['web_results']}")
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No context retrieved."
        
        response = llm.invoke([
            SystemMessage(content="""You are a knowledgeable assistant.
Answer the question based on the provided context.
If the context doesn't fully answer the question, say so and provide what you know.
Be accurate, concise, and cite which context supported your answer."""),
            HumanMessage(content=f"Question: {state['question']}\n\nContext:\n{context}"),
        ])
        
        return {"generation": response.content}
    
    # ── Routing ───────────────────────────────────────────────
    def route_after_grading(state: CRAGState) -> str:
        if state["web_search_needed"]:
            if state.get("attempts", 0) <= 1:
                return "rewrite"  # try rewriting query first
            return "web_search"   # then web search
        return "generate"
    
    # ── Build Graph ───────────────────────────────────────────
    graph = StateGraph(CRAGState)
    
    graph.add_node("retrieve",  retrieve_node)
    graph.add_node("grade",     grade_docs_node)
    graph.add_node("rewrite",   rewrite_query_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate",  generate_node)
    
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    
    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {"rewrite": "rewrite", "web_search": "web_search", "generate": "generate"}
    )
    
    graph.add_edge("rewrite", "retrieve")  # retry retrieval with better query
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    
    return graph.compile()


# ── 3B. Self-RAG ──────────────────────────────────────────────

class SelfRAGState(TypedDict):
    question: str
    needs_retrieval: bool
    retrieved_docs: List[Document]
    generation: str
    is_grounded: bool
    is_useful: bool


def build_self_rag_graph(vector_store: FAISS):
    """
    Self-RAG: the model decides whether retrieval is needed at all.
    Also checks if the generated answer is grounded and useful.
    """
    llm = ChatOpenAI(model=GPT4O_MINI, temperature=0, api_key=OPENAI_API_KEY)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    class RetrievalDecision(BaseModel):
        needs_retrieval: bool
        reason: str
    
    class GroundingCheck(BaseModel):
        is_grounded: bool       # answer supported by docs
        is_useful: bool         # answer actually helps the user
        reason: str
    
    def decide_retrieval_node(state: SelfRAGState) -> dict:
        """Decide if retrieval is needed for this question."""
        decision_llm = llm.with_structured_output(RetrievalDecision)
        
        decision = decision_llm.invoke([
            SystemMessage(content="""Decide if retrieval from a knowledge base is needed.
Retrieval IS needed: specific factual questions, technical details, recent info
Retrieval NOT needed: general knowledge, math, creative tasks, simple questions"""),
            HumanMessage(content=f"Question: {state['question']}"),
        ])
        
        icon = "📥" if decision.needs_retrieval else "🧠"
        console.print(f"  {icon} Retrieval needed: {decision.needs_retrieval} — {decision.reason}")
        
        return {"needs_retrieval": decision.needs_retrieval}
    
    def retrieve_node(state: SelfRAGState) -> dict:
        docs = retriever.invoke(state["question"])
        return {"retrieved_docs": docs}
    
    def generate_node(state: SelfRAGState) -> dict:
        context = ""
        if state.get("retrieved_docs"):
            context = "Context:\n" + "\n\n".join(
                d.page_content for d in state["retrieved_docs"]
            )
        
        response = llm.invoke([
            SystemMessage(content="Answer the question accurately and helpfully."),
            HumanMessage(content=f"{context}\n\nQuestion: {state['question']}" if context else state["question"]),
        ])
        return {"generation": response.content}
    
    def check_grounding_node(state: SelfRAGState) -> dict:
        """Check if the answer is grounded in retrieved docs."""
        if not state.get("retrieved_docs"):
            return {"is_grounded": True, "is_useful": True}  # no docs → skip check
        
        check_llm = llm.with_structured_output(GroundingCheck)
        
        doc_text = "\n".join(d.page_content for d in state["retrieved_docs"])
        
        check = check_llm.invoke([
            SystemMessage(content="""Check if the answer is:
1. Grounded: supported by the provided documents (not hallucinated)
2. Useful: actually answers the question helpfully"""),
            HumanMessage(content=f"""Question: {state['question']}
Documents: {doc_text[:500]}
Answer: {state['generation']}"""),
        ])
        
        console.print(f"  {'✅' if check.is_grounded and check.is_useful else '⚠️'} Grounded: {check.is_grounded}, Useful: {check.is_useful}")
        
        return {"is_grounded": check.is_grounded, "is_useful": check.is_useful}
    
    def route_generation(state: SelfRAGState) -> str:
        if not state.get("is_grounded") or not state.get("is_useful"):
            return "regenerate"
        return "end"
    
    graph = StateGraph(SelfRAGState)
    
    graph.add_node("decide_retrieval", decide_retrieval_node)
    graph.add_node("retrieve",         retrieve_node)
    graph.add_node("generate",         generate_node)
    graph.add_node("check",            check_grounding_node)
    
    graph.add_edge(START, "decide_retrieval")
    
    graph.add_conditional_edges(
        "decide_retrieval",
        lambda s: "retrieve" if s["needs_retrieval"] else "generate",
        {"retrieve": "retrieve", "generate": "generate"}
    )
    
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "check")
    
    graph.add_conditional_edges(
        "check",
        route_generation,
        {"regenerate": "generate", "end": END}  # retry once if not grounded
    )
    
    return graph.compile()


# ── 4. Demo ───────────────────────────────────────────────────

def run_rag_demo():
    console.print("[bold magenta]═══ 09: Agentic RAG ═══[/bold magenta]")
    
    vs = build_vector_store()
    
    questions = [
        "What is LangGraph and how does it differ from LangChain?",
        "What is Corrective RAG and why is it useful?",
        "How does vector similarity search work?",
        "What is the weather like today in Paris?",  # not in KB → web search
    ]
    
    # ── CRAG Demo ─────────────────────────────────────────────
    console.print("\n[bold cyan]=== Corrective RAG ===[/bold cyan]")
    crag = build_crag_graph(vs)
    
    for q in questions[:3]:
        console.print(f"\n[white]Q:[/white] {q}")
        result = crag.invoke({
            "question": q, "retrieved_docs": [], "graded_docs": [],
            "web_search_needed": False, "web_results": "", "generation": "", "attempts": 0,
        })
        console.print(f"[green]A:[/green] {result['generation'][:300]}...\n")
    
    # ── Self-RAG Demo ─────────────────────────────────────────
    console.print("\n[bold cyan]=== Self-RAG ===[/bold cyan]")
    self_rag = build_self_rag_graph(vs)
    
    test_questions = [
        "Explain LangGraph's checkpointing feature.",  # needs retrieval
        "What is 15 factorial?",                       # doesn't need retrieval
    ]
    
    for q in test_questions:
        console.print(f"\n[white]Q:[/white] {q}")
        result = self_rag.invoke({
            "question": q, "needs_retrieval": False, "retrieved_docs": [],
            "generation": "", "is_grounded": True, "is_useful": True,
        })
        console.print(f"[green]A:[/green] {result['generation'][:300]}...")


if __name__ == "__main__":
    run_rag_demo()
