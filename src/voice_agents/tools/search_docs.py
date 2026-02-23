from __future__ import annotations

import json

from voice_agents.tools.definitions import ToolSpec


def search_docs(query: str) -> str:
    """Stub document search returning placeholder results."""
    results = [
        {
            "title": f"Document about {query}",
            "snippet": f"This document covers key aspects of {query}. "
            "It includes best practices, common patterns, and troubleshooting steps.",
            "relevance_score": 0.92,
        },
        {
            "title": f"FAQ: {query}",
            "snippet": f"Frequently asked questions related to {query}. "
            "Covers the most common issues users encounter.",
            "relevance_score": 0.85,
        },
        {
            "title": f"Getting started with {query}",
            "snippet": f"A beginner-friendly guide to {query}. "
            "Step-by-step instructions for setup and configuration.",
            "relevance_score": 0.78,
        },
    ]
    return json.dumps({"query": query, "results": results}, indent=2)


SEARCH_DOCS_TOOL_SPEC = ToolSpec(
    name="search_docs",
    description="Search internal documentation for relevant information. Returns matching document snippets.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string",
            }
        },
        "required": ["query"],
    },
    fn=search_docs,
)
