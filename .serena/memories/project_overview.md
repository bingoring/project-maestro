# Project Maestro - Project Overview

## Purpose
AI Agent-based Game Prototyping Automation System that converts game design documents into actual Unity game prototypes using multi-agent orchestration.

## Tech Stack
- **Python**: 3.11+ (main language)
- **LangChain**: Multi-agent framework and LLM orchestration
- **FastAPI**: REST API framework
- **Unity**: 2023.2.0f1+ for game engine integration
- **Redis**: Message queueing and task management
- **PostgreSQL**: Database with SQLAlchemy ORM
- **AI Models**: OpenAI, Anthropic, Stable Diffusion

## Architecture
Multi-agent system with specialized AI agents:
- **Orchestrator**: Master agent coordinating workflow
- **Codex**: C# game code generation
- **Canvas**: Visual asset generation (Stable Diffusion)
- **Sonata**: Audio/music generation
- **Labyrinth**: Level design and gameplay
- **Builder**: Unity integration and cross-platform builds

## Current LangChain Usage
- Uses **legacy LangChain agents** (AgentExecutor, create_openai_functions_agent)
- **NOT using LCEL** (LangChain Expression Language)
- **NOT using Runnable** interface
- Traditional agent.ainvoke() pattern throughout codebase