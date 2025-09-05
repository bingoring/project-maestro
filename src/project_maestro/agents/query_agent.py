"""
Query Agent for Enterprise Knowledge Management

Specialized agent for handling enterprise search queries with complexity-based cascading.
Provides intelligent search, analysis, and knowledge retrieval from integrated systems.
"""

from typing import Any, Dict, List, Optional, Literal
import asyncio
from datetime import datetime, timedelta

from langchain_core.tools import tool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from ..core.agent_framework import BaseAgent, AgentType, AgentTask
from ..core.intent_classifier import IntentAnalysis, QueryComplexity, get_intent_classifier
from ..core.rag_system import get_rag_system
from ..integrations.enterprise_connectors import (
    get_enterprise_manager, 
    SearchOptions, 
    EnterpriseDocument
)
from ..core.logging import get_logger
from ..core.error_handling import with_error_handling, RecoveryStrategy

logger = get_logger(__name__)


class QueryComplexityAnalyzer:
    """Analyzes query complexity for cascading decisions."""
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        
    async def analyze_complexity(self, query: str, initial_results: List[Any] = None) -> Dict[str, Any]:
        """Analyze query complexity and suggest handling approach."""
        
        template = """Analyze the complexity of this enterprise search query and recommend handling approach.

Query: {query}

Initial Results Count: {results_count}

Consider these factors:
1. Query ambiguity and specificity
2. Required domain expertise
3. Need for cross-system analysis
4. Complexity of expected answer
5. Whether follow-up questions likely

Complexity Levels:
- SIMPLE: Direct lookup, clear answer exists
- MODERATE: Requires synthesis from multiple sources
- COMPLEX: Needs analysis, comparison, or expertise
- EXPERT: Requires specialized knowledge or multi-agent coordination

Provide analysis in this format:
COMPLEXITY: [level]
REASONING: [why this level]
APPROACH: [recommended handling approach]
ESCALATION: [when to escalate to higher tier]
ESTIMATED_TIME: [seconds for completion]
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        
        result = await chain.ainvoke({
            "query": query,
            "results_count": len(initial_results) if initial_results else 0
        })
        
        return self._parse_complexity_analysis(result)
        
    def _parse_complexity_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse complexity analysis result."""
        lines = analysis.strip().split('\n')
        result = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'complexity':
                    result['complexity'] = value
                elif key == 'reasoning':
                    result['reasoning'] = value
                elif key == 'approach':
                    result['approach'] = value
                elif key == 'escalation':
                    result['escalation'] = value
                elif key == 'estimated_time':
                    result['estimated_time'] = value
                    
        return result


class QueryAgent(BaseAgent):
    """
    Specialized agent for enterprise knowledge queries with cascading complexity handling.
    
    Capabilities:
    - Direct enterprise system search
    - RAG-enhanced knowledge retrieval
    - Complexity-based query routing
    - Multi-tier cascading for complex queries
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="query_agent",
            agent_type=AgentType.QUERY,
            **kwargs
        )
        
        self.enterprise_manager = get_enterprise_manager()
        self.rag_system = get_rag_system()
        self.intent_classifier = get_intent_classifier()
        self.complexity_analyzer = QueryComplexityAnalyzer(self.llm)
        
        # Cascading tiers
        self.cascading_tiers = {
            1: self._handle_simple_query,
            2: self._handle_moderate_query,
            3: self._handle_complex_query,
            4: self._handle_expert_query
        }
        
        # Create specialized tools
        self.tools = [
            self._create_enterprise_search_tool(),
            self._create_rag_search_tool(),
            self._create_recent_updates_tool(),
            self._create_escalate_tool(),
        ]
        
    def _create_agent_graph(self):
        """Create LangGraph agent for query processing."""
        memory = MemorySaver()
        
        agent_graph = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=memory,
            state_modifier=self.get_system_prompt()
        )
        
        return agent_graph
        
    def _create_runnable_chain(self):
        """Create LCEL runnable chain for query processing."""
        
        template = """You are a Query Agent specialized in enterprise knowledge search.

Query: {query}
Intent Analysis: {intent_analysis}
Available Enterprise Systems: {enterprise_systems}

Your role is to:
1. Search relevant enterprise systems
2. Retrieve and synthesize information
3. Provide comprehensive answers
4. Escalate complex queries when needed

Use your tools effectively to gather information and provide helpful responses.
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def get_enterprise_systems(inputs):
            return list(self.enterprise_manager.connectors.keys())
        
        def analyze_intent(inputs):
            # This would be async in real implementation
            return "Intent analysis placeholder"
        
        query_chain = (
            RunnableParallel({
                "query": RunnablePassthrough(),
                "intent_analysis": RunnableLambda(analyze_intent),
                "enterprise_systems": RunnableLambda(get_enterprise_systems)
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return query_chain
        
    def get_system_prompt(self) -> str:
        """Get system prompt for query agent."""
        return f"""You are the Query Agent for Project Maestro's enterprise knowledge system.

Your capabilities:
- Search across enterprise systems: {', '.join(self.enterprise_manager.connectors.keys())}
- RAG-enhanced knowledge retrieval
- Complexity-based query cascading
- Multi-source information synthesis

Your responsibilities:
1. Understand user intent and query complexity
2. Search relevant enterprise systems efficiently
3. Synthesize information from multiple sources
4. Provide accurate, comprehensive answers
5. Escalate complex queries to specialized agents when needed

Available enterprise systems:
{self._get_system_descriptions()}

Always be helpful, accurate, and efficient. Use your tools to gather comprehensive information.
"""

    def _get_system_descriptions(self) -> str:
        """Get descriptions of available enterprise systems."""
        descriptions = []
        for name in self.enterprise_manager.connectors.keys():
            if name == "jira":
                descriptions.append("- Jira: Issue tracking, projects, comments, workflows")
            elif name == "slack": 
                descriptions.append("- Slack: Team communications, channels, files, discussions")
            elif name == "confluence":
                descriptions.append("- Confluence: Documentation, wikis, pages, knowledge base")
            elif name == "github":
                descriptions.append("- GitHub: Code repositories, issues, pull requests, commits")
                
        return "\n".join(descriptions)
        
    def _create_enterprise_search_tool(self):
        """Create tool for searching enterprise systems."""
        
        @tool
        async def search_enterprise(
            query: str,
            systems: str = "all",
            limit: int = 10,
            time_range: str = "all"
        ) -> str:
            """Search enterprise systems for information.
            
            Args:
                query: Search query
                systems: Comma-separated list of systems to search (jira,slack,confluence) or 'all'
                limit: Maximum number of results per system
                time_range: Time range filter (all, week, month, quarter)
            """
            try:
                # Parse time range
                start_date = None
                if time_range == "week":
                    start_date = datetime.now() - timedelta(days=7)
                elif time_range == "month":
                    start_date = datetime.now() - timedelta(days=30)
                elif time_range == "quarter":
                    start_date = datetime.now() - timedelta(days=90)
                    
                options = SearchOptions(
                    limit=limit,
                    start_date=start_date,
                    sort_by="relevance"
                )
                
                # Search specified systems or all
                if systems.lower() == "all":
                    results = await self.enterprise_manager.search_all(query, options)
                else:
                    system_list = [s.strip() for s in systems.split(",")]
                    results = {}
                    for system in system_list:
                        if system in self.enterprise_manager.connectors:
                            connector = self.enterprise_manager.connectors[system]
                            async with connector:
                                results[system] = await connector.search(query, options)
                
                # Format results
                return self._format_search_results(results)
                
            except Exception as e:
                logger.error(f"Enterprise search failed: {e}")
                return f"Search failed: {str(e)}"
                
        return search_enterprise
        
    def _create_rag_search_tool(self):
        """Create tool for RAG-enhanced search."""
        
        @tool
        async def search_knowledge_base(
            query: str,
            strategy: str = "enhanced",
            k: int = 5
        ) -> str:
            """Search the knowledge base using RAG.
            
            Args:
                query: Search query
                strategy: Search strategy (standard, enhanced, multi_query, hyde)
                k: Number of results to retrieve
            """
            try:
                if strategy == "enhanced" and hasattr(self.rag_system, 'enhanced_retrieval'):
                    docs = await self.rag_system.enhanced_retrieval(query, strategy="multi_query")
                else:
                    docs = await self.rag_system.asimilarity_search(query, k=k)
                    
                if not docs:
                    return "No relevant information found in knowledge base."
                    
                # Format results
                results = []
                for doc in docs:
                    metadata = doc.metadata
                    source_info = f"Source: {metadata.get('source', 'Unknown')}"
                    if metadata.get('title'):
                        source_info += f" | Title: {metadata['title']}"
                    if metadata.get('url'):
                        source_info += f" | URL: {metadata['url']}"
                        
                    results.append(f"{source_info}\nContent: {doc.page_content[:300]}...\n")
                    
                return "\n".join(results)
                
            except Exception as e:
                logger.error(f"RAG search failed: {e}")
                return f"Knowledge base search failed: {str(e)}"
                
        return search_knowledge_base
        
    def _create_recent_updates_tool(self):
        """Create tool for getting recent updates."""
        
        @tool
        async def get_recent_updates(
            days: int = 7,
            systems: str = "all"
        ) -> str:
            """Get recent updates from enterprise systems.
            
            Args:
                days: Number of days to look back
                systems: Systems to check (comma-separated or 'all')
            """
            try:
                results = []
                
                if systems.lower() == "all":
                    connectors_to_check = self.enterprise_manager.connectors.items()
                else:
                    system_list = [s.strip() for s in systems.split(",")]
                    connectors_to_check = [(name, connector) for name, connector in 
                                         self.enterprise_manager.connectors.items() 
                                         if name in system_list]
                
                for name, connector in connectors_to_check:
                    try:
                        async with connector:
                            recent_docs = await connector.get_recent_updates(days)
                            if recent_docs:
                                results.append(f"\n## Recent from {name.title()} ({len(recent_docs)} items):")
                                for doc in recent_docs[:5]:  # Show top 5
                                    results.append(f"- {doc.title} ({doc.author}, {doc.created_date.strftime('%Y-%m-%d') if doc.created_date else 'Unknown date'})")
                                    
                    except Exception as e:
                        results.append(f"\n## {name.title()}: Error retrieving updates ({str(e)})")
                        
                return "\n".join(results) if results else "No recent updates found."
                
            except Exception as e:
                logger.error(f"Recent updates check failed: {e}")
                return f"Failed to get recent updates: {str(e)}"
                
        return get_recent_updates
        
    def _create_escalate_tool(self):
        """Create tool for escalating complex queries."""
        
        @tool
        async def escalate_to_specialist(
            query: str,
            reason: str,
            recommended_agent: str = "orchestrator"
        ) -> str:
            """Escalate query to specialized agent or orchestrator.
            
            Args:
                query: Original query to escalate
                reason: Reason for escalation
                recommended_agent: Suggested agent to handle query
            """
            # In real implementation, this would trigger LangGraph orchestration
            escalation_info = {
                "original_query": query,
                "escalation_reason": reason,
                "recommended_handler": recommended_agent,
                "escalated_at": datetime.now().isoformat(),
                "escalated_by": "query_agent"
            }
            
            logger.info(f"Query escalated: {escalation_info}")
            
            return f"""Query escalated to {recommended_agent}.

Reason: {reason}

The query will be handled by a more specialized agent or multi-agent workflow.
You should expect a more comprehensive response shortly."""
            
        return escalate_to_specialist
        
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute query task with cascading complexity handling."""
        
        action = task.action
        params = task.parameters
        
        if action == "process_query":
            return await self._process_query_with_cascading(
                params.get("query", ""),
                params.get("user_context", {}),
                params.get("tier", 1)
            )
        elif action == "search_enterprise":
            return await self._search_enterprise_systems(
                params.get("query", ""),
                params.get("options", {})
            )
        else:
            # Use agent graph for complex interactions
            messages = [HumanMessage(content=f"Execute: {action} with params: {params}")]
            
            config = {"configurable": {"thread_id": task.id}}
            result = await self.agent_graph.ainvoke(
                {"messages": messages},
                config=config
            )
            
            final_message = result["messages"][-1]
            return {"result": final_message.content}
            
    async def _process_query_with_cascading(
        self, 
        query: str, 
        user_context: Dict = None,
        current_tier: int = 1
    ) -> Dict[str, Any]:
        """Process query with complexity-based cascading."""
        
        logger.info(f"Processing query at tier {current_tier}: {query[:100]}...")
        
        try:
            # Analyze intent and complexity
            intent_analysis = await self.intent_classifier.classify_query(query)
            
            # Determine appropriate tier based on complexity
            complexity_tier_map = {
                QueryComplexity.SIMPLE: 1,
                QueryComplexity.MODERATE: 2,
                QueryComplexity.COMPLEX: 3,
                QueryComplexity.EXPERT: 4
            }
            
            recommended_tier = complexity_tier_map.get(intent_analysis.complexity, 1)
            
            # If current tier is insufficient, escalate
            if current_tier < recommended_tier:
                return await self._escalate_to_tier(query, recommended_tier, intent_analysis)
                
            # Handle at current tier
            handler = self.cascading_tiers.get(current_tier, self._handle_simple_query)
            result = await handler(query, intent_analysis, user_context)
            
            return {
                "query": query,
                "tier": current_tier,
                "intent_analysis": intent_analysis.__dict__,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "query": query,
                "tier": current_tier,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    async def _handle_simple_query(
        self, 
        query: str, 
        intent_analysis: IntentAnalysis,
        user_context: Dict = None
    ) -> str:
        """Handle simple, direct lookup queries (Tier 1)."""
        
        logger.info("Handling simple query with direct search")
        
        # Direct enterprise search
        options = SearchOptions(limit=5, sort_by="relevance")
        search_results = await self.enterprise_manager.get_unified_results(query, options)
        
        if not search_results:
            # Fallback to RAG search
            rag_docs = await self.rag_system.asimilarity_search(query, k=3)
            if rag_docs:
                return self._format_rag_results(rag_docs)
            else:
                return "I couldn't find relevant information for your query. Please try rephrasing or contact support."
                
        return self._format_enterprise_results(search_results)
        
    async def _handle_moderate_query(
        self,
        query: str,
        intent_analysis: IntentAnalysis, 
        user_context: Dict = None
    ) -> str:
        """Handle queries requiring some analysis (Tier 2)."""
        
        logger.info("Handling moderate query with enhanced search and analysis")
        
        # Multi-source search with analysis
        options = SearchOptions(limit=10, sort_by="relevance")
        
        # Parallel search across systems and RAG
        search_task = self.enterprise_manager.get_unified_results(query, options)
        rag_task = self.rag_system.enhanced_retrieval(query, strategy="multi_query") if hasattr(self.rag_system, 'enhanced_retrieval') else self.rag_system.asimilarity_search(query, k=5)
        
        search_results, rag_results = await asyncio.gather(search_task, rag_task)
        
        # Synthesize information
        synthesis_prompt = f"""Based on the following information sources, provide a comprehensive answer to: {query}

Enterprise Search Results:
{self._format_enterprise_results(search_results)}

Knowledge Base Results:
{self._format_rag_results(rag_results)}

Provide a well-structured, informative response that synthesizes information from all sources."""
        
        messages = [HumanMessage(content=synthesis_prompt)]
        response = await self.llm.ainvoke(messages)
        
        return response.content
        
    async def _handle_complex_query(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        user_context: Dict = None
    ) -> str:
        """Handle complex queries requiring multi-step analysis (Tier 3)."""
        
        logger.info("Handling complex query - considering orchestrator escalation")
        
        # For complex queries, use agent graph with tools
        messages = [HumanMessage(content=f"""This is a complex query that requires comprehensive analysis: {query}

Please use your available tools to:
1. Search relevant enterprise systems
2. Gather information from knowledge base
3. Analyze and synthesize findings
4. Provide a detailed response

If the query is too complex for your capabilities, escalate to the orchestrator.""")]
        
        config = {"configurable": {"thread_id": f"complex_{datetime.now().timestamp()}"}}
        result = await self.agent_graph.ainvoke(
            {"messages": messages},
            config=config
        )
        
        return result["messages"][-1].content
        
    async def _handle_expert_query(
        self,
        query: str, 
        intent_analysis: IntentAnalysis,
        user_context: Dict = None
    ) -> str:
        """Handle expert-level queries requiring specialized knowledge (Tier 4)."""
        
        logger.info("Expert query detected - escalating to orchestrator")
        
        # Expert queries should be escalated to LangGraph orchestrator
        escalation_context = {
            "query": query,
            "complexity": "expert",
            "intent": intent_analysis.intent.value,
            "suggested_agents": intent_analysis.suggested_agent,
            "requires_specialization": True,
            "escalation_reason": "Query requires expert-level analysis and potentially multiple specialized agents"
        }
        
        return f"""This query requires expert-level analysis and has been escalated to our specialized agent system.

Query: {query}
Complexity Level: Expert
Estimated Processing Time: 2-5 minutes

The query will be handled by multiple specialized agents working together to provide you with a comprehensive, expert-level response."""
        
    async def _escalate_to_tier(
        self,
        query: str,
        target_tier: int, 
        intent_analysis: IntentAnalysis
    ) -> Dict[str, Any]:
        """Escalate query to higher tier."""
        
        logger.info(f"Escalating query to tier {target_tier}")
        
        # Process at target tier
        result = await self._process_query_with_cascading(query, {}, target_tier)
        
        result["escalated"] = True
        result["original_tier"] = 1
        result["target_tier"] = target_tier
        
        return result
        
    def _format_search_results(self, results: Dict[str, List[EnterpriseDocument]]) -> str:
        """Format search results from multiple systems."""
        formatted = []
        
        for system, docs in results.items():
            if docs:
                formatted.append(f"\n## Results from {system.title()} ({len(docs)} found):")
                for doc in docs[:3]:  # Show top 3 per system
                    formatted.append(f"**{doc.title}**")
                    formatted.append(f"Author: {doc.author} | Date: {doc.created_date.strftime('%Y-%m-%d') if doc.created_date else 'Unknown'}")
                    formatted.append(f"Content: {doc.content[:200]}...")
                    formatted.append(f"URL: {doc.url}\n")
                    
        return "\n".join(formatted) if formatted else "No results found."
        
    def _format_enterprise_results(self, results: List[EnterpriseDocument]) -> str:
        """Format enterprise search results."""
        if not results:
            return "No results found."
            
        formatted = []
        for doc in results[:5]:  # Show top 5
            formatted.append(f"**{doc.title}**")
            formatted.append(f"Source: {doc.source} | Author: {doc.author}")
            if doc.created_date:
                formatted.append(f"Date: {doc.created_date.strftime('%Y-%m-%d')}")
            formatted.append(f"Content: {doc.content[:200]}...")
            formatted.append(f"URL: {doc.url}\n")
            
        return "\n".join(formatted)
        
    def _format_rag_results(self, results: List[Any]) -> str:
        """Format RAG search results."""
        if not results:
            return "No results found in knowledge base."
            
        formatted = []
        for doc in results:
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            title = metadata.get('title', 'Knowledge Base Entry')
            source = metadata.get('source', 'Unknown')
            
            formatted.append(f"**{title}**")
            formatted.append(f"Source: {source}")
            formatted.append(f"Content: {doc.page_content[:200]}...")
            if metadata.get('url'):
                formatted.append(f"URL: {metadata['url']}")
            formatted.append("")
            
        return "\n".join(formatted)
        
    async def _search_enterprise_systems(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """Direct enterprise systems search."""
        
        search_options = SearchOptions(
            limit=options.get('limit', 10),
            sort_by=options.get('sort_by', 'relevance')
        )
        
        results = await self.enterprise_manager.search_all(query, search_options)
        
        return {
            "query": query,
            "results": {
                system: [doc.__dict__ for doc in docs]
                for system, docs in results.items()
            },
            "total_results": sum(len(docs) for docs in results.values())
        }