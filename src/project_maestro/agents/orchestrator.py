"""Orchestrator agent - the master agent for workflow management."""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel as PydanticModel

from ..core.agent_framework import BaseAgent, AgentType, AgentTask, AgentStatus
from ..core.message_queue import EventType, publish_event
from ..core.logging import get_logger
from ..core.gdd_parser import GameDesignDocumentParser, GameComplexityAnalyzer, ParsedGameDesign


class ProjectSpec(PydanticModel):
    """Project specification parsed from game design document."""
    id: str
    title: str
    description: str
    genre: str
    platform: str = "mobile"
    art_style: str
    gameplay_mechanics: List[str]
    characters: List[Dict[str, Any]]
    environments: List[Dict[str, Any]]
    sounds: List[Dict[str, Any]]
    levels: List[Dict[str, Any]]
    technical_requirements: Dict[str, Any]
    estimated_complexity: int = Field(ge=1, le=10)  # 1-10 scale


class WorkflowStep(PydanticModel):
    """Represents a step in the project workflow."""
    id: str
    name: str
    agent_type: AgentType
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str] = []  # IDs of steps that must complete first
    priority: int = 5
    estimated_duration: int = 300  # seconds
    status: str = "pending"


class ProjectWorkflow(PydanticModel):
    """Complete workflow for a project."""
    project_id: str
    steps: List[WorkflowStep]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "created"
    progress: float = 0.0


class EnhancedWorkflowPlanner:
    """Enhanced workflow planner that uses complexity analysis."""
    
    def __init__(self, llm):
        self.llm = llm
        self.logger = get_logger("enhanced_workflow_planner")
        self.complexity_analyzer = GameComplexityAnalyzer()
        
    async def create_enhanced_workflow(self, parsed_design: ParsedGameDesign) -> ProjectWorkflow:
        """Create an enhanced execution workflow with complexity analysis."""
        
        # Analyze development effort
        effort_analysis = self.complexity_analyzer.analyze_development_effort(parsed_design)
        
        # Extract asset requirements
        parser = GameDesignDocumentParser(self.llm)
        asset_requirements = parser.extract_asset_requirements(parsed_design)
        
        workflow_steps = []
        step_counter = 0
        
        # Create steps based on asset requirements
        # Code generation steps
        for req in asset_requirements["code"]:
            step_counter += 1
            workflow_steps.append(WorkflowStep(
                id=f"step_{step_counter}",
                name=f"Generate {req['type']}: {req['name']}",
                agent_type=AgentType.CODEX,
                action="generate_code",
                parameters={
                    "requirement": req,
                    "project_design": parsed_design.dict()
                },
                priority=9 if req.get("complexity") == "complex" else 7,
                estimated_duration=effort_analysis["time_estimates"]["code_generation"]
            ))
            
        # Visual asset generation steps
        for asset in asset_requirements["visual"]:
            step_counter += 1
            workflow_steps.append(WorkflowStep(
                id=f"step_{step_counter}",
                name=f"Generate visual asset: {asset['name']}",
                agent_type=AgentType.CANVAS,
                action="generate_visual_asset",
                parameters={
                    "asset": asset,
                    "project_design": parsed_design.dict()
                },
                priority=8,
                estimated_duration=effort_analysis["time_estimates"]["asset_creation"] // max(len(asset_requirements["visual"]), 1)
            ))
            
        # Audio generation steps
        for audio in asset_requirements["audio"]:
            step_counter += 1
            workflow_steps.append(WorkflowStep(
                id=f"step_{step_counter}",
                name=f"Generate audio: {audio['name']}",
                agent_type=AgentType.SONATA,
                action="generate_audio",
                parameters={
                    "audio": audio,
                    "project_design": parsed_design.dict()
                },
                priority=6,
                estimated_duration=effort_analysis["time_estimates"]["asset_creation"] // max(len(asset_requirements["audio"]), 1)
            ))
            
        # Level design steps (depend on code mechanics)
        code_step_ids = [step.id for step in workflow_steps if step.agent_type == AgentType.CODEX]
        
        for level in asset_requirements["level"]:
            step_counter += 1
            workflow_steps.append(WorkflowStep(
                id=f"step_{step_counter}",
                name=f"Design level: {level['name']}",
                agent_type=AgentType.LABYRINTH,
                action="design_level",
                parameters={
                    "level": level,
                    "project_design": parsed_design.dict()
                },
                dependencies=code_step_ids,  # Levels depend on game mechanics
                priority=7,
                estimated_duration=effort_analysis["time_estimates"]["level_design"] // max(len(asset_requirements["level"]), 1)
            ))
            
        # Final build step (depends on everything)
        step_counter += 1
        all_previous_steps = [step.id for step in workflow_steps]
        
        workflow_steps.append(WorkflowStep(
            id=f"step_{step_counter}",
            name="Build final game prototype",
            agent_type=AgentType.BUILDER,
            action="build_game",
            parameters={
                "project_design": parsed_design.dict(),
                "effort_analysis": effort_analysis
            },
            dependencies=all_previous_steps,
            priority=10,
            estimated_duration=effort_analysis["time_estimates"]["build"]
        ))
        
        # Create project ID from metadata
        project_id = str(uuid.uuid4())
        
        workflow = ProjectWorkflow(
            project_id=project_id,
            steps=workflow_steps,
            created_at=datetime.now(),
            status="created"
        )
        
        self.logger.info(
            "Created enhanced project workflow",
            project_id=project_id,
            total_steps=len(workflow_steps),
            estimated_duration=sum(step.estimated_duration for step in workflow_steps),
            complexity_score=parsed_design.metadata.get("complexity_score", 0.5),
            development_risks=len(effort_analysis.get("development_risks", []))
        )
        
        return workflow


class WorkflowPlanner:
    """Plans the execution workflow for a project."""
    
    def __init__(self, llm):
        self.llm = llm
        self.logger = get_logger("workflow_planner")
        
    async def create_workflow(self, project_spec: ProjectSpec) -> ProjectWorkflow:
        """Create an execution workflow for the project."""
        
        workflow_steps = []
        step_counter = 0
        
        # Step 1: Character asset generation
        for character in project_spec.characters:
            step_counter += 1
            workflow_steps.append(WorkflowStep(
                id=f"step_{step_counter}",
                name=f"Generate {character['name']} character assets",
                agent_type=AgentType.CANVAS,
                action="generate_character_sprites",
                parameters={
                    "character": character,
                    "art_style": project_spec.art_style,
                    "project_id": project_spec.id
                },
                priority=8,
                estimated_duration=600
            ))
            
        # Step 2: Environment asset generation
        for environment in project_spec.environments:
            step_counter += 1
            workflow_steps.append(WorkflowStep(
                id=f"step_{step_counter}",
                name=f"Generate {environment['name']} environment assets",
                agent_type=AgentType.CANVAS,
                action="generate_environment_assets",
                parameters={
                    "environment": environment,
                    "art_style": project_spec.art_style,
                    "project_id": project_spec.id
                },
                priority=7,
                estimated_duration=800
            ))
            
        # Step 3: Sound generation
        for sound in project_spec.sounds:
            step_counter += 1
            workflow_steps.append(WorkflowStep(
                id=f"step_{step_counter}",
                name=f"Generate {sound['type']} sound: {sound['name']}",
                agent_type=AgentType.SONATA,
                action="generate_sound",
                parameters={
                    "sound": sound,
                    "project_id": project_spec.id
                },
                priority=6,
                estimated_duration=400
            ))
            
        # Step 4: Core game mechanics code generation
        mechanics_steps = []
        for i, mechanic in enumerate(project_spec.gameplay_mechanics):
            step_counter += 1
            step_id = f"step_{step_counter}"
            mechanics_steps.append(step_id)
            
            workflow_steps.append(WorkflowStep(
                id=step_id,
                name=f"Implement {mechanic} mechanic",
                agent_type=AgentType.CODEX,
                action="generate_gameplay_code",
                parameters={
                    "mechanic": mechanic,
                    "project_spec": project_spec.dict(),
                    "project_id": project_spec.id
                },
                priority=9,
                estimated_duration=500
            ))
            
        # Step 5: Level generation (depends on mechanics)
        level_steps = []
        for level in project_spec.levels:
            step_counter += 1
            step_id = f"step_{step_counter}"
            level_steps.append(step_id)
            
            workflow_steps.append(WorkflowStep(
                id=step_id,
                name=f"Generate level: {level['name']}",
                agent_type=AgentType.LABYRINTH,
                action="generate_level",
                parameters={
                    "level": level,
                    "gameplay_mechanics": project_spec.gameplay_mechanics,
                    "project_id": project_spec.id
                },
                dependencies=mechanics_steps,  # Levels depend on mechanics
                priority=7,
                estimated_duration=300
            ))
            
        # Step 6: Final build (depends on everything)
        step_counter += 1
        all_previous_steps = [step.id for step in workflow_steps]
        
        workflow_steps.append(WorkflowStep(
            id=f"step_{step_counter}",
            name="Build final game prototype",
            agent_type=AgentType.BUILDER,
            action="build_game",
            parameters={
                "project_spec": project_spec.dict(),
                "project_id": project_spec.id
            },
            dependencies=all_previous_steps,
            priority=10,
            estimated_duration=1200
        ))
        
        workflow = ProjectWorkflow(
            project_id=project_spec.id,
            steps=workflow_steps,
            created_at=datetime.now(),
            status="created"
        )
        
        self.logger.info(
            "Created project workflow",
            project_id=project_spec.id,
            total_steps=len(workflow_steps),
            estimated_duration=sum(step.estimated_duration for step in workflow_steps)
        )
        
        return workflow


class OrchestratorTools:
    """Tools available to the Orchestrator agent."""
    
    @staticmethod
    def create_parse_document_tool(orchestrator: "OrchestratorAgent") -> BaseTool:
        """Tool for parsing game design documents."""
        
        class ParseDocumentInput(BaseModel):
            document: str = Field(description="The game design document text to parse")
            
        async def parse_document(document: str) -> str:
            try:
                parsed_design = orchestrator.parser.parse_document(document)
                orchestrator.current_parsed_design = parsed_design
                
                # Analyze complexity
                effort_analysis = orchestrator.complexity_analyzer.analyze_development_effort(parsed_design)
                orchestrator.current_effort_analysis = effort_analysis
                
                # Publish project created event
                await publish_event(
                    EventType.PROJECT_CREATED,
                    "orchestrator",
                    {
                        "project_id": str(uuid.uuid4()),
                        "title": parsed_design.metadata.get("title", "Unknown"),
                        "complexity": parsed_design.metadata.get("complexity_score", 0.5),
                        "estimated_duration": sum(effort_analysis["time_estimates"].values()),
                        "development_risks": len(effort_analysis.get("development_risks", []))
                    }
                )
                
                return f"Successfully parsed project: {parsed_design.metadata.get('title', 'Unknown')} (Complexity: {parsed_design.metadata.get('complexity_score', 0.5):.2f})"
            except Exception as e:
                return f"Error parsing document: {str(e)}"
                
        return StructuredTool.from_function(
            func=parse_document,
            name="parse_document",
            description="Parse a game design document into structured project specification",
            args_schema=ParseDocumentInput
        )
        
    @staticmethod
    def create_plan_workflow_tool(orchestrator: "OrchestratorAgent") -> BaseTool:
        """Tool for creating project workflows."""
        
        async def plan_workflow() -> str:
            try:
                if not orchestrator.current_parsed_design:
                    return "Error: No parsed design available. Parse a document first."
                    
                workflow = await orchestrator.enhanced_planner.create_enhanced_workflow(
                    orchestrator.current_parsed_design
                )
                orchestrator.current_workflow = workflow
                
                estimated_time = sum(step.estimated_duration for step in workflow.steps)
                risk_count = len(orchestrator.current_effort_analysis.get("development_risks", []))
                
                return f"Created enhanced workflow with {len(workflow.steps)} steps for project {workflow.project_id}. Estimated time: {estimated_time//60} minutes. Risks identified: {risk_count}"
            except Exception as e:
                return f"Error creating workflow: {str(e)}"
                
        return StructuredTool.from_function(
            func=plan_workflow,
            name="plan_workflow",
            description="Create an execution workflow for the current project",
        )
        
    @staticmethod 
    def create_execute_workflow_tool(orchestrator: "OrchestratorAgent") -> BaseTool:
        """Tool for executing project workflows."""
        
        async def execute_workflow() -> str:
            try:
                if not orchestrator.current_workflow:
                    return "Error: No workflow available. Create a workflow first."
                    
                await orchestrator.execute_workflow()
                
                return f"Started workflow execution for project {orchestrator.current_workflow.project_id}"
            except Exception as e:
                return f"Error executing workflow: {str(e)}"
                
        return StructuredTool.from_function(
            func=execute_workflow,
            name="execute_workflow", 
            description="Execute the current project workflow",
        )


class OrchestratorAgent(BaseAgent):
    """The master orchestrator agent that manages the entire workflow."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="maestro_orchestrator",
            agent_type=AgentType.ORCHESTRATOR,
            **kwargs
        )
        
        self.parser = GameDesignDocumentParser(self.llm)
        self.enhanced_planner = EnhancedWorkflowPlanner(self.llm)
        self.complexity_analyzer = GameComplexityAnalyzer()
        
        # Current project state
        self.current_project_spec: Optional[ProjectSpec] = None
        self.current_parsed_design: Optional[ParsedGameDesign] = None
        self.current_effort_analysis: Optional[Dict[str, Any]] = None
        self.current_workflow: Optional[ProjectWorkflow] = None
        self.executing_steps: Set[str] = set()
        self.completed_steps: Set[str] = set()
        
        # Create tools
        self.tools = [
            OrchestratorTools.create_parse_document_tool(self),
            OrchestratorTools.create_plan_workflow_tool(self),
            OrchestratorTools.create_execute_workflow_tool(self),
        ]
        
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor for the orchestrator."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[self.callback_handler]
        )
        
    def get_system_prompt(self) -> str:
        """Get the system prompt for the orchestrator."""
        return """
        You are the Maestro Orchestrator, the master AI agent responsible for converting game design 
        documents into playable game prototypes. You coordinate a team of specialist AI agents to 
        create assets and code.

        Your responsibilities:
        1. Parse game design documents into structured specifications
        2. Create detailed execution workflows
        3. Coordinate specialist agents (Codex, Canvas, Sonata, Labyrinth)
        4. Monitor progress and handle errors
        5. Ensure quality and consistency across all generated assets

        Your specialist agents:
        - Codex: Generates C# game code and scripts
        - Canvas: Creates art assets (sprites, backgrounds, UI)
        - Sonata: Generates music and sound effects
        - Labyrinth: Designs levels and gameplay progression
        - Builder: Integrates all assets and builds the final game

        Always follow this workflow:
        1. Parse the game design document first
        2. Create a detailed workflow plan
        3. Execute the workflow by coordinating agents
        4. Monitor progress and handle any issues

        Be thorough, professional, and ensure the final game matches the original vision.
        """
        
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a task using the orchestrator."""
        
        action = task.action
        params = task.parameters
        
        if action == "process_game_document":
            return await self._process_game_document(params.get("document", ""))
        elif action == "get_project_status":
            return await self._get_project_status()
        else:
            # Use the agent executor for complex reasoning
            result = await self.agent_executor.ainvoke({
                "input": f"Execute action: {action} with parameters: {params}"
            })
            return {"result": result["output"]}
            
    async def _process_game_document(self, document: str) -> Dict[str, Any]:
        """Process a complete game design document."""
        
        self.log_agent_action("process_game_document", "started")
        
        try:
            # Step 1: Parse document with enhanced parser
            parsed_design = self.parser.parse_document(document)
            self.current_parsed_design = parsed_design
            
            # Step 2: Analyze complexity and effort
            effort_analysis = self.complexity_analyzer.analyze_development_effort(parsed_design)
            self.current_effort_analysis = effort_analysis
            
            # Step 3: Create enhanced workflow
            workflow = await self.enhanced_planner.create_enhanced_workflow(parsed_design)
            self.current_workflow = workflow
            
            # Step 4: Start workflow execution
            await self.execute_workflow()
            
            return {
                "project_id": workflow.project_id,
                "title": parsed_design.metadata.get("title", "Unknown"),
                "workflow_steps": len(workflow.steps),
                "complexity_score": parsed_design.metadata.get("complexity_score", 0.5),
                "estimated_duration": sum(effort_analysis["time_estimates"].values()),
                "development_risks": len(effort_analysis.get("development_risks", [])),
                "status": "workflow_started"
            }
            
        except Exception as e:
            self.log_agent_error("process_game_document", e)
            raise
            
    async def execute_workflow(self):
        """Execute the current workflow."""
        
        if not self.current_workflow:
            raise ValueError("No workflow to execute")
            
        self.log_agent_action("execute_workflow", "started",
                            project_id=self.current_workflow.project_id)
        
        workflow = self.current_workflow
        workflow.started_at = datetime.now()
        workflow.status = "running"
        
        # Execute steps based on dependencies
        remaining_steps = {step.id: step for step in workflow.steps}
        
        while remaining_steps:
            # Find steps that can be executed (all dependencies completed)
            ready_steps = []
            
            for step_id, step in remaining_steps.items():
                if step_id not in self.executing_steps:
                    dependencies_met = all(
                        dep_id in self.completed_steps 
                        for dep_id in step.dependencies
                    )
                    if dependencies_met:
                        ready_steps.append(step)
                        
            if not ready_steps:
                # Check if we're just waiting for executing steps to complete
                if self.executing_steps:
                    await asyncio.sleep(1)
                    continue
                else:
                    # Deadlock - some dependencies can't be satisfied
                    self.log_agent_error(
                        "execute_workflow",
                        Exception("Workflow deadlock - unresolvable dependencies")
                    )
                    break
                    
            # Execute ready steps
            for step in ready_steps:
                asyncio.create_task(self._execute_workflow_step(step))
                self.executing_steps.add(step.id)
                
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            
        # Wait for all executing steps to complete
        while self.executing_steps:
            await asyncio.sleep(1)
            
        # Mark workflow as completed
        workflow.completed_at = datetime.now()
        workflow.status = "completed"
        workflow.progress = 1.0
        
        # Publish completion event
        await publish_event(
            EventType.PROJECT_COMPLETED,
            "orchestrator",
            {
                "project_id": workflow.project_id,
                "steps_completed": len(self.completed_steps),
                "total_steps": len(workflow.steps)
            }
        )
        
        self.log_agent_success("execute_workflow",
                             {"project_id": workflow.project_id})
                             
    async def _execute_workflow_step(self, step: WorkflowStep):
        """Execute a single workflow step."""
        
        try:
            self.log_agent_action("execute_step", "started",
                                step_id=step.id, step_name=step.name)
            
            # Create agent task
            task = AgentTask(
                agent_type=step.agent_type,
                action=step.action,
                parameters=step.parameters,
                priority=step.priority,
                timeout=step.estimated_duration
            )
            
            # Publish task created event
            await publish_event(
                EventType.TASK_CREATED,
                "orchestrator",
                {
                    "task_id": task.id,
                    "agent_type": step.agent_type.value,
                    "action": step.action,
                    "step_id": step.id
                }
            )
            
            step.status = "completed"
            
        except Exception as e:
            step.status = "failed"
            self.log_agent_error("execute_step", e,
                               step_id=step.id, step_name=step.name)
        finally:
            self.executing_steps.discard(step.id)
            self.completed_steps.add(step.id)
            
    async def _get_project_status(self) -> Dict[str, Any]:
        """Get current project status."""
        
        if not self.current_parsed_design:
            return {"status": "no_project"}
            
        status = {
            "project_id": self.current_workflow.project_id if self.current_workflow else "unknown",
            "title": self.current_parsed_design.metadata.get("title", "Unknown"),
            "complexity_score": self.current_parsed_design.metadata.get("complexity_score", 0.5),
            "status": self.current_workflow.status if self.current_workflow else "planning"
        }
        
        if self.current_workflow:
            completed = len(self.completed_steps)
            total = len(self.current_workflow.steps)
            status.update({
                "progress": completed / total if total > 0 else 0,
                "completed_steps": completed,
                "total_steps": total,
                "executing_steps": len(self.executing_steps)
            })
            
        if self.current_effort_analysis:
            status.update({
                "estimated_total_time": sum(self.current_effort_analysis["time_estimates"].values()),
                "development_risks": len(self.current_effort_analysis.get("development_risks", [])),
                "recommendations": self.current_effort_analysis.get("recommendations", [])
            })
            
        return status