"""
LangGraph Multi-Agent Orchestration System for Project Maestro

This module implements a sophisticated multi-agent system using LangGraph for game development
automation. It provides orchestration, handoffs, and state management for specialized agents.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated, Sequence
from dataclasses import dataclass
from enum import Enum
import asyncio
import uuid
from datetime import datetime

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .agent_framework import BaseAgent, AgentType, AgentTask, AgentStatus
from .logging import get_logger
from .error_handling import with_error_handling, RecoveryStrategy
from .monitoring import agent_monitor, metrics_collector


logger = get_logger(__name__)


class AgentHandoffType(str, Enum):
    """Types of agent handoffs in the multi-agent system."""
    DIRECT = "direct"          # Direct handoff to specific agent
    SUPERVISOR = "supervisor"  # Handoff back to supervisor
    PARALLEL = "parallel"      # Parallel execution with multiple agents
    SEQUENTIAL = "sequential"  # Sequential workflow through multiple agents


class MaestroState(TypedDict):
    """Enhanced state management for Project Maestro multi-agent system."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_agent: str
    task_context: Dict[str, Any]
    game_design_doc: Optional[Dict[str, Any]]
    assets_generated: Dict[str, List[str]]  # Track generated assets by type
    code_artifacts: Dict[str, str]  # Track generated code by component
    build_status: Dict[str, Any]
    workflow_stage: str
    handoff_history: List[Dict[str, Any]]
    execution_metadata: Dict[str, Any]


@dataclass
class AgentCapability:
    """Defines what an agent can do and when to use it."""
    agent_name: str
    specializations: List[str]
    prerequisites: List[str] = None
    outputs: List[str] = None
    confidence_threshold: float = 0.7


class LangGraphOrchestrator:
    """
    LangGraph-powered orchestrator for Project Maestro multi-agent system.
    
    Manages complex game development workflows through intelligent agent coordination,
    handoffs, and state management using LangGraph's graph-based approach.
    """
    
    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        llm: BaseLanguageModel,
        memory_saver: Optional[MemorySaver] = None
    ):
        self.agents = agents
        self.llm = llm
        self.memory_saver = memory_saver or MemorySaver()
        
        # Initialize intent classifier for enterprise workflows
        from ..core.intent_classifier import IntentClassifier
        self.intent_classifier = IntentClassifier(llm)
        
        # Define agent capabilities for intelligent routing
        self.agent_capabilities = {
            "orchestrator": AgentCapability(
                agent_name="orchestrator",
                specializations=["workflow_coordination", "task_delegation", "progress_tracking"],
                outputs=["task_assignments", "workflow_plans"]
            ),
            "codex": AgentCapability(
                agent_name="codex",
                specializations=["csharp_code", "unity_scripting", "gameplay_logic"],
                prerequisites=["game_design_doc"],
                outputs=["csharp_scripts", "unity_components"]
            ),
            "canvas": AgentCapability(
                agent_name="canvas", 
                specializations=["visual_assets", "ui_design", "textures", "sprites"],
                outputs=["images", "ui_components", "visual_assets"]
            ),
            "sonata": AgentCapability(
                agent_name="sonata",
                specializations=["audio", "music", "sound_effects"],
                outputs=["audio_files", "music_compositions"]
            ),
            "labyrinth": AgentCapability(
                agent_name="labyrinth",
                specializations=["level_design", "gameplay_mechanics", "balancing"],
                prerequisites=["game_design_doc"],
                outputs=["level_data", "gameplay_rules"]
            ),
            "builder": AgentCapability(
                agent_name="builder",
                specializations=["unity_integration", "builds", "deployment"],
                prerequisites=["csharp_scripts", "visual_assets"],
                outputs=["unity_project", "game_builds"]
            ),
            "query": AgentCapability(
                agent_name="query",
                specializations=["enterprise_search", "knowledge_retrieval", "information_synthesis"],
                outputs=["search_results", "knowledge_summaries", "information_analysis"]
            )
        }
        
        # Build the main orchestration graph
        self.graph = self._build_orchestration_graph()
        
    def _build_orchestration_graph(self) -> StateGraph:
        """Build the main LangGraph orchestration graph with enterprise support."""
        logger.info("Building enhanced LangGraph orchestration graph with enterprise support")
        
        # Initialize the graph with enhanced state
        graph = StateGraph(MaestroState)
        
        # Add agent nodes
        for agent_name, agent in self.agents.items():
            graph.add_node(agent_name, self._create_agent_node(agent_name, agent))
            
        # Add supervisor node for intelligent routing
        graph.add_node("supervisor", self._create_supervisor_node())
        
        # Add workflow coordination nodes
        graph.add_node("workflow_planner", self._create_workflow_planner_node())
        graph.add_node("progress_tracker", self._create_progress_tracker_node())
        graph.add_node("quality_gate", self._create_quality_gate_node())
        
        # ====== NEW: Enterprise workflow nodes ======
        graph.add_node("intent_analyzer", self._create_intent_analyzer_node())
        graph.add_node("enterprise_router", self._create_enterprise_router_node())
        graph.add_node("knowledge_analyzer", self._create_knowledge_analyzer_node())
        
        # ====== Define graph edges and conditional routing ======
        
        # Start with intent analysis for all queries
        graph.add_edge(START, "intent_analyzer")
        
        # Route from intent analyzer based on query type
        graph.add_conditional_edges(
            "intent_analyzer",
            self._route_from_intent_analysis,
            ["enterprise_router", "workflow_planner", "supervisor"]
        )
        
        # Enterprise routing for knowledge queries
        graph.add_conditional_edges(
            "enterprise_router", 
            self._route_enterprise_query,
            ["query", "knowledge_analyzer", "supervisor", END]
        )
        
        # Knowledge analyzer for moderate complexity queries
        graph.add_edge("knowledge_analyzer", "supervisor")
        
        # Traditional workflow routing (preserved)
        graph.add_edge("workflow_planner", "supervisor")
        
        # Add conditional edges for agent routing from supervisor
        graph.add_conditional_edges(
            "supervisor",
            self._route_to_agent,
            list(self.agents.keys()) + ["progress_tracker", "quality_gate", END]
        )
        
        # All agents report back to supervisor
        for agent_name in self.agents.keys():
            graph.add_edge(agent_name, "supervisor")
            
        # Progress tracking and quality gates
        graph.add_edge("progress_tracker", "supervisor")
        graph.add_conditional_edges(
            "quality_gate",
            self._quality_gate_decision,
            ["supervisor", END]
        )
        
        # Compile with memory for persistence
        compiled_graph = graph.compile(checkpointer=self.memory_saver)
        
        logger.info("Enhanced LangGraph orchestration graph compiled successfully")
        return compiled_graph

        
    def _create_intent_analyzer_node(self):
        """Create intent analysis node for routing queries appropriately."""
        async def intent_analyzer(state: MaestroState) -> Command:
            logger.info("Analyzing query intent and complexity")
            
            # Get the user's query
            messages = state.get("messages", [])
            if not messages:
                return Command(
                    update={"workflow_stage": "error", "messages": [AIMessage(content="No input provided")]},
                    goto=END
                )
                
            user_query = messages[0].content if messages else ""
            
            try:
                # Analyze intent and complexity
                intent_analysis = await self.intent_classifier.classify_query(user_query)
                
                # Store analysis in state
                updated_state = dict(state)
                updated_state["task_context"] = {
                    "original_query": user_query,
                    "intent_analysis": intent_analysis.__dict__,
                    "routing_decision": "pending"
                }
                updated_state["workflow_stage"] = "intent_analyzed"
                
                analysis_message = AIMessage(
                    content=f"Intent: {intent_analysis.intent.value}, Complexity: {intent_analysis.complexity.value}, Confidence: {intent_analysis.confidence:.2f}",
                    name="intent_analyzer"
                )
                
                return Command(
                    update={
                        **updated_state,
                        "messages": [analysis_message]
                    },
                    goto="enterprise_router"  # Will be properly routed by conditional edge
                )
                
            except Exception as e:
                logger.error(f"Intent analysis failed: {e}")
                return Command(
                    update={
                        "workflow_stage": "error",
                        "messages": [AIMessage(content=f"Intent analysis failed: {str(e)}")]
                    },
                    goto=END
                )
                
        return intent_analyzer
        
    def _create_enterprise_router_node(self):
        """Create enterprise query router for complexity-based routing."""
        async def enterprise_router(state: MaestroState) -> Command:
            logger.info("Routing enterprise query based on complexity")
            
            task_context = state.get("task_context", {})
            intent_analysis = task_context.get("intent_analysis", {})
            
            if not intent_analysis:
                logger.warning("No intent analysis found, defaulting to supervisor")
                return Command(goto="supervisor")
            
            complexity = intent_analysis.get("complexity", "simple")
            intent = intent_analysis.get("intent", "general_chat")
            
            # Update routing decision
            updated_context = dict(task_context)
            updated_context["routing_decision"] = f"enterprise_query_{complexity}"
            
            routing_message = AIMessage(
                content=f"Routing enterprise query (complexity: {complexity})",
                name="enterprise_router"
            )
            
            return Command(
                update={
                    "task_context": updated_context,
                    "messages": [routing_message]
                },
                goto="pending"  # Will be determined by conditional routing
            )
            
        return enterprise_router
        
    def _create_knowledge_analyzer_node(self):
        """Create knowledge analyzer for moderate complexity queries."""
        async def knowledge_analyzer(state: MaestroState) -> Command:
            logger.info("Performing knowledge analysis for moderate complexity query")
            
            task_context = state.get("task_context", {})
            original_query = task_context.get("original_query", "")
            
            # Perform enhanced analysis
            analysis_result = f"""Knowledge Analysis for: {original_query}

This query requires moderate complexity handling involving:
1. Multi-source information gathering
2. Context synthesis from enterprise systems
3. Analysis and structured response

Preparing to delegate to Query Agent with enhanced capabilities."""
            
            analysis_message = AIMessage(
                content=analysis_result,
                name="knowledge_analyzer"
            )
            
            # Enhance task context for query agent
            enhanced_context = dict(task_context)
            enhanced_context["analysis_level"] = "moderate"
            enhanced_context["recommended_tier"] = 2
            enhanced_context["requires_synthesis"] = True
            
            return Command(
                update={
                    "task_context": enhanced_context,
                    "workflow_stage": "knowledge_analyzed",
                    "messages": [analysis_message]
                },
                goto="supervisor"
            )
            
        return knowledge_analyzer
        
    def _route_from_intent_analysis(self, state: MaestroState) -> str:
        """Route based on intent analysis results."""
        task_context = state.get("task_context", {})
        intent_analysis = task_context.get("intent_analysis", {})
        
        if not intent_analysis:
            logger.warning("No intent analysis found, routing to supervisor")
            return "supervisor"
            
        intent = intent_analysis.get("intent", "general_chat")
        complexity = intent_analysis.get("complexity", "simple")
        
        logger.info(f"Routing decision: intent={intent}, complexity={complexity}")
        
        # Route game development queries to traditional workflow
        if intent == "game_development":
            return "workflow_planner"
            
        # Route enterprise search and knowledge queries
        elif intent in ["enterprise_search", "project_info", "technical_support"]:
            return "enterprise_router"
            
        # Route workflow automation to supervisor (complex coordination)
        elif intent in ["workflow_automation", "complex_analysis"]:
            return "supervisor"
            
        # Default routing for general chat and unknown intents
        else:
            return "supervisor"
            
    def _route_enterprise_query(self, state: MaestroState) -> str:
        """Route enterprise queries based on complexity."""
        task_context = state.get("task_context", {})
        intent_analysis = task_context.get("intent_analysis", {})
        
        if not intent_analysis:
            return "supervisor"
            
        complexity = intent_analysis.get("complexity", "simple")
        confidence = intent_analysis.get("confidence", 0.5)
        
        logger.info(f"Enterprise routing: complexity={complexity}, confidence={confidence}")
        
        # Simple queries go directly to query agent
        if complexity == "simple" and confidence > 0.7:
            return "query"
            
        # Moderate complexity goes through knowledge analyzer first
        elif complexity == "moderate":
            return "knowledge_analyzer"
            
        # Complex and expert queries need supervisor orchestration
        elif complexity in ["complex", "expert"]:
            return "supervisor"
            
        # Low confidence queries get handled by query agent with fallback
        else:
            return "query"
        
    def _create_agent_node(self, agent_name: str, agent: BaseAgent):
        """Create a LangGraph node for a specific agent."""
        async def agent_node(state: MaestroState) -> Command:
            logger.info(f"Executing agent: {agent_name}")
            
            # Extract current task from state
            current_message = state["messages"][-1]
            task_context = state.get("task_context", {})
            
            # Create agent task
            task = AgentTask(
                id=str(uuid.uuid4()),
                agent_type=agent.agent_type,
                action=f"process_request_{agent_name}",
                parameters=task_context,
                priority=1
            )
            
            try:
                # Execute task using the agent
                completed_task = await agent.process_task(task)
                
                # Prepare response message
                if completed_task.status == AgentStatus.COMPLETED:
                    response_content = f"Agent {agent_name} completed task: {completed_task.result}"
                    
                    # Update state with agent outputs
                    updated_state = dict(state)
                    updated_state["current_agent"] = agent_name
                    
                    # Store agent-specific outputs
                    if agent_name == "codex" and completed_task.result:
                        updated_state.setdefault("code_artifacts", {})[agent_name] = completed_task.result
                    elif agent_name == "canvas" and completed_task.result:
                        updated_state.setdefault("assets_generated", {}).setdefault("visual", []).append(completed_task.result)
                    elif agent_name == "sonata" and completed_task.result:
                        updated_state.setdefault("assets_generated", {}).setdefault("audio", []).append(completed_task.result)
                        
                else:
                    response_content = f"Agent {agent_name} failed: {completed_task.error}"
                    
            except Exception as e:
                logger.error(f"Agent {agent_name} execution failed: {e}")
                response_content = f"Agent {agent_name} encountered error: {str(e)}"
                updated_state = dict(state)
                
            # Record handoff history
            updated_state.setdefault("handoff_history", []).append({
                "from_agent": state.get("current_agent", "supervisor"),
                "to_agent": agent_name,
                "timestamp": datetime.now().isoformat(),
                "task_context": task_context
            })
            
            return Command(
                update={
                    **updated_state,
                    "messages": [AIMessage(content=response_content, name=agent_name)]
                },
                goto="supervisor"
            )
            
        return agent_node
        
    def _create_supervisor_node(self):
        """Create the supervisor node for intelligent agent routing."""
        
        # Create handoff tools for each agent
        handoff_tools = []
        for agent_name, capability in self.agent_capabilities.items():
            if agent_name != "supervisor":
                handoff_tools.append(self._create_handoff_tool(agent_name, capability))
                
        # Add workflow control tools
        handoff_tools.extend([
            self._create_progress_tracker_tool(),
            self._create_quality_gate_tool(),
            self._create_completion_tool()
        ])
        
        # Create supervisor agent with routing capabilities
        supervisor_agent = create_react_agent(
            self.llm,
            handoff_tools,
            state_modifier=self._get_supervisor_prompt()
        )
        
        async def supervisor_node(state: MaestroState) -> Command:
            logger.info("Supervisor analyzing task and routing")
            
            # Add context to supervisor
            enhanced_messages = list(state["messages"])
            
            # Add workflow context
            context_message = self._create_context_message(state)
            enhanced_messages.append(context_message)
            
            # Get supervisor decision
            response = await supervisor_agent.ainvoke({
                "messages": enhanced_messages
            })
            
            # Extract last message and check for tool calls
            last_message = response["messages"][-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # Tool calls will be handled by the tools directly
                return Command(
                    update={"messages": response["messages"]},
                    goto="supervisor"  # Stay in supervisor for tool execution
                )
            else:
                # No tool calls - likely a completion message
                return Command(
                    update={"messages": response["messages"]},
                    goto=END
                )
                
        return supervisor_node
        
    def _create_handoff_tool(self, agent_name: str, capability: AgentCapability):
        """Create a handoff tool for transferring to specific agent."""
        
        @tool
        def transfer_to_agent(task_description: str, context: Dict[str, Any] = None) -> Command:
            f"""Transfer task to {agent_name} agent.
            
            Use this when you need: {', '.join(capability.specializations)}
            
            Args:
                task_description: Description of the task for the agent
                context: Additional context for the task
            """
            logger.info(f"Transferring to {agent_name}: {task_description}")
            
            return Command(
                goto=agent_name,
                update={
                    "task_context": {
                        "description": task_description,
                        "context": context or {},
                        "requested_by": "supervisor",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )
            
        # Dynamically set tool name and description
        transfer_to_agent.name = f"transfer_to_{agent_name}"
        transfer_to_agent.__doc__ = f"""Transfer task to {agent_name} agent.
        
        Use this when you need: {', '.join(capability.specializations)}
        Prerequisites: {', '.join(capability.prerequisites) if capability.prerequisites else 'None'}
        Outputs: {', '.join(capability.outputs) if capability.outputs else 'Various'}
        """
        
        return transfer_to_agent
        
    def _create_progress_tracker_tool(self):
        """Create tool for progress tracking."""
        @tool
        def check_progress(stage: str = None) -> Command:
            """Check current workflow progress and status."""
            return Command(
                goto="progress_tracker",
                update={"task_context": {"check_stage": stage}}
            )
        return check_progress
        
    def _create_quality_gate_tool(self):
        """Create tool for quality validation."""
        @tool
        def validate_quality(component: str = None) -> Command:
            """Validate quality of generated components."""
            return Command(
                goto="quality_gate", 
                update={"task_context": {"validate_component": component}}
            )
        return validate_quality
        
    def _create_completion_tool(self):
        """Create tool for workflow completion."""
        @tool  
        def complete_workflow(summary: str) -> Command:
            """Complete the current workflow with summary."""
            return Command(
                goto=END,
                update={
                    "workflow_stage": "completed",
                    "execution_metadata": {
                        "completion_summary": summary,
                        "completed_at": datetime.now().isoformat()
                    }
                }
            )
        return complete_workflow
        
    def _create_workflow_planner_node(self):
        """Create workflow planning node."""
        async def workflow_planner(state: MaestroState) -> Command:
            logger.info("Planning workflow execution")
            
            # Analyze the incoming request
            messages = state.get("messages", [])
            if not messages:
                return Command(
                    update={"workflow_stage": "error", "messages": [AIMessage(content="No input provided")]},
                    goto=END
                )
                
            initial_request = messages[0].content if messages else ""
            
            # Create workflow plan
            workflow_plan = {
                "request": initial_request,
                "planned_stages": self._generate_workflow_stages(initial_request),
                "estimated_duration": "TBD",
                "required_agents": self._identify_required_agents(initial_request)
            }
            
            return Command(
                update={
                    "workflow_stage": "planning_complete",
                    "task_context": workflow_plan,
                    "messages": [AIMessage(content=f"Workflow planned: {workflow_plan['planned_stages']}")]
                },
                goto="supervisor"
            )
            
        return workflow_planner
        
    def _create_progress_tracker_node(self):
        """Create progress tracking node."""
        async def progress_tracker(state: MaestroState) -> Command:
            logger.info("Tracking workflow progress")
            
            # Analyze current state
            completed_agents = set()
            for handoff in state.get("handoff_history", []):
                completed_agents.add(handoff["to_agent"])
                
            progress_report = {
                "completed_agents": list(completed_agents),
                "assets_generated": len(state.get("assets_generated", {})),
                "code_artifacts": len(state.get("code_artifacts", {})),
                "current_stage": state.get("workflow_stage", "unknown"),
                "progress_percentage": self._calculate_progress(state)
            }
            
            return Command(
                update={
                    "execution_metadata": {
                        **state.get("execution_metadata", {}),
                        "progress_report": progress_report
                    },
                    "messages": [AIMessage(content=f"Progress update: {progress_report}")]
                },
                goto="supervisor"
            )
            
        return progress_tracker
        
    def _create_quality_gate_node(self):
        """Create quality validation node."""
        async def quality_gate(state: MaestroState) -> Command:
            logger.info("Performing quality validation")
            
            # Validate generated artifacts
            quality_checks = {
                "code_quality": self._validate_code_artifacts(state.get("code_artifacts", {})),
                "asset_quality": self._validate_generated_assets(state.get("assets_generated", {})),
                "workflow_completeness": self._validate_workflow_completeness(state)
            }
            
            overall_quality = all(quality_checks.values())
            
            return Command(
                update={
                    "execution_metadata": {
                        **state.get("execution_metadata", {}),
                        "quality_checks": quality_checks
                    },
                    "messages": [AIMessage(content=f"Quality validation: {'PASSED' if overall_quality else 'FAILED'}")]
                },
                goto="supervisor" if overall_quality else "supervisor"  # Continue or retry
            )
            
        return quality_gate
        
    def _route_to_agent(self, state: MaestroState) -> str:
        """Route to appropriate agent based on state."""
        # Check for completion conditions
        if state.get("workflow_stage") == "completed":
            return END
            
        # Check last message for tool calls (handled by LangGraph)
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # Let LangGraph handle tool routing
                return "supervisor"
                
        # Default routing logic
        current_stage = state.get("workflow_stage", "initial")
        
        if current_stage == "planning_complete":
            return "supervisor"
        elif current_stage in ["needs_progress_check", "progress_requested"]:
            return "progress_tracker"
        elif current_stage in ["needs_validation", "quality_check_requested"]:
            return "quality_gate"
        else:
            return "supervisor"
            
    def _quality_gate_decision(self, state: MaestroState) -> str:
        """Decide next step after quality gate."""
        quality_checks = state.get("execution_metadata", {}).get("quality_checks", {})
        if all(quality_checks.values()):
            return END
        else:
            return "supervisor"  # Continue workflow for fixes
            
    def _get_supervisor_prompt(self) -> str:
        """Get the supervisor system prompt."""
        return """You are the Project Maestro Supervisor, orchestrating a multi-agent game development system.

Your role is to:
1. Analyze incoming game development requests
2. Route tasks to appropriate specialized agents
3. Monitor progress and coordinate handoffs
4. Ensure quality and completeness

Available Agents and Their Specializations:
- orchestrator: Workflow coordination, task delegation, progress tracking
- codex: C# code generation, Unity scripting, gameplay logic  
- canvas: Visual assets, UI design, textures, sprites
- sonata: Audio generation, music composition, sound effects
- labyrinth: Level design, gameplay mechanics, balancing
- builder: Unity integration, builds, deployment

Workflow Guidelines:
- Start with workflow planning
- Route tasks based on agent specializations
- Check prerequisites before delegating (e.g., game design doc for code generation)
- Track progress and validate quality
- Coordinate handoffs between agents
- Complete workflow when all requirements are satisfied

Always explain your routing decisions and provide clear task descriptions to agents.
"""

    def _create_context_message(self, state: MaestroState) -> HumanMessage:
        """Create context message for supervisor."""
        context_info = {
            "workflow_stage": state.get("workflow_stage", "unknown"),
            "completed_agents": [h["to_agent"] for h in state.get("handoff_history", [])],
            "available_assets": list(state.get("assets_generated", {}).keys()),
            "available_code": list(state.get("code_artifacts", {}).keys()),
            "task_context": state.get("task_context", {})
        }
        
        context_text = f"Current workflow context: {context_info}"
        return HumanMessage(content=context_text, name="system")
        
    def _generate_workflow_stages(self, request: str) -> List[str]:
        """Generate workflow stages based on request."""
        # Basic workflow stages for game development
        base_stages = ["analysis", "design", "implementation", "testing", "deployment"]
        
        # Customize based on request content
        if "audio" in request.lower() or "music" in request.lower():
            base_stages.insert(-2, "audio_generation")
        if "visual" in request.lower() or "art" in request.lower():
            base_stages.insert(-2, "visual_generation")
        if "level" in request.lower() or "gameplay" in request.lower():
            base_stages.insert(-2, "level_design")
            
        return base_stages
        
    def _identify_required_agents(self, request: str) -> List[str]:
        """Identify which agents are needed for the request."""
        required = ["orchestrator"]  # Always needed
        
        request_lower = request.lower()
        
        if any(term in request_lower for term in ["code", "script", "programming", "c#"]):
            required.append("codex")
        if any(term in request_lower for term in ["visual", "art", "sprite", "texture", "ui"]):
            required.append("canvas")
        if any(term in request_lower for term in ["audio", "music", "sound"]):
            required.append("sonata")
        if any(term in request_lower for term in ["level", "gameplay", "mechanic"]):
            required.append("labyrinth")
        if any(term in request_lower for term in ["build", "deploy", "unity", "compile"]):
            required.append("builder")
            
        return required
        
    def _calculate_progress(self, state: MaestroState) -> float:
        """Calculate workflow progress percentage."""
        total_agents = len(self.agents)
        completed_agents = len(set(h["to_agent"] for h in state.get("handoff_history", [])))
        
        if total_agents == 0:
            return 0.0
            
        return min(100.0, (completed_agents / total_agents) * 100.0)
        
    def _validate_code_artifacts(self, code_artifacts: Dict[str, str]) -> bool:
        """Validate generated code artifacts."""
        # Basic validation - check if code exists and is not empty
        return bool(code_artifacts) and all(
            isinstance(code, str) and len(code.strip()) > 0 
            for code in code_artifacts.values()
        )
        
    def _validate_generated_assets(self, assets: Dict[str, List[str]]) -> bool:
        """Validate generated assets."""
        # Basic validation - check if assets exist
        return bool(assets) and any(asset_list for asset_list in assets.values())
        
    def _validate_workflow_completeness(self, state: MaestroState) -> bool:
        """Validate overall workflow completeness."""
        required_elements = ["task_context", "handoff_history"]
        return all(state.get(element) for element in required_elements)
        
    async def execute_workflow(
        self, 
        request: str,
        thread_id: Optional[str] = None,
        stream_mode: str = "updates"
    ):
        """Execute a complete workflow using the LangGraph orchestration."""
        if not thread_id:
            thread_id = str(uuid.uuid4())
            
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"Starting workflow execution for thread {thread_id}")
        
        initial_state = {
            "messages": [HumanMessage(content=request)],
            "workflow_stage": "initiated",
            "execution_metadata": {
                "started_at": datetime.now().isoformat(),
                "thread_id": thread_id
            }
        }
        
        try:
            async for chunk in self.graph.astream(
                initial_state,
                config=config,
                stream_mode=stream_mode
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            yield {
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "thread_id": thread_id
                }
            }

        
    async def execute_enterprise_query(
        self,
        query: str,
        user_context: Dict = None,
        thread_id: Optional[str] = None,
        stream_mode: str = "updates"
    ):
        """Execute an enterprise knowledge query using the enhanced orchestration."""
        if not thread_id:
            thread_id = f"enterprise_{uuid.uuid4()}"
            
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"Starting enterprise query execution for thread {thread_id}")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "workflow_stage": "enterprise_query_initiated",
            "current_agent": "intent_analyzer",
            "task_context": {
                "query_type": "enterprise",
                "user_context": user_context or {},
                "initiated_at": datetime.now().isoformat()
            },
            "execution_metadata": {
                "started_at": datetime.now().isoformat(),
                "thread_id": thread_id,
                "workflow_type": "enterprise_query"
            }
        }
        
        try:
            async for chunk in self.graph.astream(
                initial_state,
                config=config,
                stream_mode=stream_mode
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Enterprise query execution failed: {e}")
            yield {
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "thread_id": thread_id,
                    "workflow_type": "enterprise_query"
                }
            }
            
    async def execute_hybrid_workflow(
        self,
        request: str,
        workflow_type: Literal["game_development", "enterprise_query", "hybrid"] = "hybrid",
        thread_id: Optional[str] = None,
        stream_mode: str = "updates"
    ):
        """Execute a hybrid workflow that can handle both game development and enterprise queries."""
        if not thread_id:
            thread_id = f"hybrid_{uuid.uuid4()}"
            
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"Starting hybrid workflow execution for thread {thread_id}")
        
        initial_state = {
            "messages": [HumanMessage(content=request)],
            "workflow_stage": "hybrid_workflow_initiated",
            "current_agent": "intent_analyzer",
            "task_context": {
                "workflow_type": workflow_type,
                "request": request,
                "initiated_at": datetime.now().isoformat()
            },
            "execution_metadata": {
                "started_at": datetime.now().isoformat(),
                "thread_id": thread_id,
                "workflow_type": "hybrid"
            }
        }
        
        try:
            async for chunk in self.graph.astream(
                initial_state,
                config=config,
                stream_mode=stream_mode
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Hybrid workflow execution failed: {e}")
            yield {
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "thread_id": thread_id,
                    "workflow_type": "hybrid"
                }
            }
            
    async def get_enterprise_capabilities(self) -> Dict[str, Any]:
        """Get information about enterprise integration capabilities."""
        from ..integrations.enterprise_connectors import get_enterprise_manager
        
        enterprise_manager = get_enterprise_manager()
        
        # Test connectivity to all systems
        connection_status = await enterprise_manager.test_all_connections()
        
        capabilities = {
            "supported_systems": list(enterprise_manager.connectors.keys()),
            "connection_status": connection_status,
            "query_types_supported": [
                "enterprise_search",
                "project_info", 
                "technical_support",
                "workflow_automation",
                "complex_analysis"
            ],
            "complexity_levels": ["simple", "moderate", "complex", "expert"],
            "cascading_enabled": True,
            "intent_classification": True,
            "multi_agent_coordination": True,
            "rag_integration": True
        }
        
        return capabilities
            
    async def get_workflow_state(self, thread_id: str) -> Dict[str, Any]:
        """Get current state of a workflow."""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            state = await self.graph.aget_state(config)
            return state.values if state else {}
        except Exception as e:
            logger.error(f"Failed to get workflow state: {e}")
            return {"error": str(e)}
            
    def visualize_graph(self) -> str:
        """Get Mermaid diagram of the orchestration graph."""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            logger.error(f"Failed to generate graph visualization: {e}")
            return f"Error generating visualization: {e}"