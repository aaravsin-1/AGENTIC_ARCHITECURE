You never pick just one pattern. A real trading agent would use:

LangGraph state machine (file 02) as the backbone
Supervisor or hierarchical (03/07) for sub-agents
Memory (08) for trade history
Advanced tools (10) for API integrations
Production patterns (13) for streaming + error handling

The files I gave you are building blocks, not complete apps. A real project looks like this:
trading_agent/
├── main.py              ← entry point, assembles everything
├── config.py            ← API keys, settings
├── state.py             ← your TypedDict state definition
├── agents/
│   ├── supervisor.py    ← routing brain
│   ├── analyst.py       ← market analysis sub-agent
│   ├── risk_manager.py  ← risk check sub-agent
│   └── executor.py      ← trade execution sub-agent
├── tools/
│   ├── market_data.py   ← price feeds, indicators
│   ├── broker_api.py    ← place orders
│   └── news_search.py   ← sentiment tools
├── memory/
│   └── trade_memory.py  ← vector store for past trades
└── graph.py             ← assembles all nodes into LangGraph
