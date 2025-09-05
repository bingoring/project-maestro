"""
Intent Classification System for Project Maestro

Analyzes user queries to determine intent and route to appropriate agents:
- Game development queries -> Game development workflow
- Enterprise knowledge queries -> Query Agent
- Complex multi-domain queries -> LangGraph orchestration
"""

from typing import Dict, List, Optional, Literal, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda

from .logging import get_logger
from .config import settings

logger = get_logger(__name__)


class QueryIntent(str, Enum):
    """Types of user query intents."""
    GAME_DEVELOPMENT = "game_development"
    ENTERPRISE_SEARCH = "enterprise_search"
    PROJECT_INFO = "project_info"
    TECHNICAL_SUPPORT = "technical_support"
    WORKFLOW_AUTOMATION = "workflow_automation"
    COMPLEX_ANALYSIS = "complex_analysis"
    GENERAL_CHAT = "general_chat"


class QueryComplexity(str, Enum):
    """Query complexity levels for cascading."""
    SIMPLE = "simple"        # Direct search/lookup
    MODERATE = "moderate"    # Requires some processing
    COMPLEX = "complex"      # Multi-step analysis
    EXPERT = "expert"        # Requires specialist agents


@dataclass
class IntentAnalysis:
    """Result of intent analysis."""
    intent: QueryIntent
    complexity: QueryComplexity
    confidence: float
    keywords: List[str]
    suggested_agent: str
    requires_langgraph: bool
    data_sources: List[str]  # Which enterprise systems to query
    metadata: Dict[str, any] = None


class IntentClassificationModel(BaseModel):
    """Pydantic model for structured intent classification output."""
    intent: str = Field(description="Primary intent of the query")
    complexity: str = Field(description="Complexity level: simple, moderate, complex, expert")
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)
    keywords: List[str] = Field(description="Key terms and phrases from the query")
    reasoning: str = Field(description="Explanation of the classification decision")
    data_sources: List[str] = Field(description="Relevant data sources: jira, slack, confluence, etc.")


class IntentClassifier:
    """
    Advanced intent classification system for routing queries appropriately.
    
    Analyzes user queries to determine:
    - Primary intent (game dev, enterprise search, etc.)
    - Complexity level for appropriate routing
    - Required data sources and agents
    """
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        
        # Keyword patterns for quick classification
        self.intent_patterns = {
            QueryIntent.GAME_DEVELOPMENT: {
                "keywords": [
                    "game", "unity", "c#", "sprite", "level", "player", "enemy", 
                    "gameplay", "mechanics", "prototype", "build", "deploy",
                    "audio", "music", "sound", "visual", "art", "texture"
                ],
                "phrases": [
                    "create game", "develop game", "game design", "unity project",
                    "game prototype", "game mechanics", "level design"
                ]
            },
            QueryIntent.ENTERPRISE_SEARCH: {
                "keywords": [
                    "find", "search", "lookup", "where", "who", "what", "when",
                    "project", "team", "deadline", "status", "issue", "ticket",
                    "confluence", "jira", "slack", "document", "wiki"
                ],
                "phrases": [
                    "find information", "search for", "look up", "who is working",
                    "project status", "team member", "deadline for"
                ]
            },
            QueryIntent.WORKFLOW_AUTOMATION: {
                "keywords": [
                    "automate", "workflow", "process", "integrate", "connect",
                    "sync", "export", "import", "batch", "schedule"
                ],
                "phrases": [
                    "automate process", "create workflow", "integrate with",
                    "sync data", "batch process"
                ]
            }
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            QueryComplexity.SIMPLE: [
                "find", "what is", "who is", "when", "where", "show me"
            ],
            QueryComplexity.MODERATE: [
                "analyze", "compare", "summarize", "explain", "how to"
            ],
            QueryComplexity.COMPLEX: [
                "create", "design", "implement", "optimize", "integrate"
            ],
            QueryComplexity.EXPERT: [
                "architect", "scale", "performance", "security", "enterprise"
            ]
        }
        
        self._classification_chain = None
        
    @property
    def classification_chain(self):
        """Get or create the intent classification chain."""
        if not self._classification_chain:
            self._classification_chain = self._create_classification_chain()
        return self._classification_chain
        
    def _create_classification_chain(self):
        """Create LLM-powered intent classification chain."""
        
        template = """You are an expert intent classifier for an enterprise AI system that handles both game development and general business queries.

Analyze the user query and classify it according to these categories:

**Intent Categories:**
1. game_development - Creating games, Unity projects, game assets, gameplay mechanics
2. enterprise_search - Finding information in company systems (Jira, Slack, Confluence)
3. project_info - Project status, team information, deadlines, resources
4. technical_support - Technical help, troubleshooting, how-to questions
5. workflow_automation - Process automation, integrations, data sync
6. complex_analysis - Multi-step analysis requiring specialized knowledge
7. general_chat - Casual conversation, greetings, general questions

**Complexity Levels:**
- simple: Direct lookup or search (e.g., "What is the status of PROJ-123?")
- moderate: Requires processing/analysis (e.g., "Summarize last week's sprint")
- complex: Multi-step workflow (e.g., "Create integration between Jira and Slack")
- expert: Requires specialized agents (e.g., "Design microservices architecture")

**Data Sources to consider:**
- jira: Issue tracking, project management
- slack: Team communications, channels
- confluence: Documentation, wikis
- github: Code repositories, commits
- calendar: Meetings, schedules
- hr_systems: Employee information
- none: No external data needed

User Query: {query}

Provide your analysis in the following JSON format:
{{
    "intent": "category_name",
    "complexity": "complexity_level", 
    "confidence": 0.95,
    "keywords": ["key", "terms"],
    "reasoning": "Explanation of classification",
    "data_sources": ["relevant", "sources"]
}}
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        classification_chain = (
            prompt 
            | self.llm
            | StrOutputParser()
            | RunnableLambda(self._parse_classification_result)
        )
        
        return classification_chain
        
    def _parse_classification_result(self, result: str) -> IntentAnalysis:
        """Parse LLM classification result into structured format."""
        try:
            import json
            
            # Extract JSON from LLM response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                # Fallback parsing
                parsed = {
                    "intent": "general_chat",
                    "complexity": "simple", 
                    "confidence": 0.5,
                    "keywords": [],
                    "reasoning": "Failed to parse LLM response",
                    "data_sources": ["none"]
                }
                
            # Convert to IntentAnalysis
            intent = QueryIntent(parsed.get("intent", "general_chat"))
            complexity = QueryComplexity(parsed.get("complexity", "simple"))
            
            # Determine routing
            suggested_agent = self._determine_suggested_agent(intent, complexity)
            requires_langgraph = self._requires_langgraph(intent, complexity)
            
            return IntentAnalysis(
                intent=intent,
                complexity=complexity,
                confidence=parsed.get("confidence", 0.5),
                keywords=parsed.get("keywords", []),
                suggested_agent=suggested_agent,
                requires_langgraph=requires_langgraph,
                data_sources=parsed.get("data_sources", ["none"]),
                metadata={
                    "reasoning": parsed.get("reasoning", ""),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse classification result: {e}")
            return IntentAnalysis(
                intent=QueryIntent.GENERAL_CHAT,
                complexity=QueryComplexity.SIMPLE,
                confidence=0.3,
                keywords=[],
                suggested_agent="query_agent",
                requires_langgraph=False,
                data_sources=["none"],
                metadata={"error": str(e)}
            )
            
    def _determine_suggested_agent(self, intent: QueryIntent, complexity: QueryComplexity) -> str:
        """Determine which agent should handle this query."""
        
        if intent == QueryIntent.GAME_DEVELOPMENT:
            if complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
                return "orchestrator"  # Use full game dev workflow
            else:
                return "codex"  # Simple game dev queries
                
        elif intent == QueryIntent.ENTERPRISE_SEARCH:
            if complexity == QueryComplexity.SIMPLE:
                return "query_agent"  # Direct search
            else:
                return "knowledge_analyst"  # Analysis required
                
        elif intent == QueryIntent.WORKFLOW_AUTOMATION:
            return "orchestrator"  # Always use orchestrator for automation
            
        elif intent == QueryIntent.COMPLEX_ANALYSIS:
            return "orchestrator"  # Complex multi-agent coordination
            
        else:
            return "query_agent"  # Default to query agent
            
    def _requires_langgraph(self, intent: QueryIntent, complexity: QueryComplexity) -> bool:
        """Determine if query requires LangGraph orchestration."""
        
        # Always use LangGraph for complex workflows
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            return True
            
        # Use LangGraph for multi-domain intents
        if intent in [QueryIntent.GAME_DEVELOPMENT, QueryIntent.WORKFLOW_AUTOMATION]:
            return True
            
        # Simple queries can use direct agent routing
        return False
        
    async def classify_query(self, query: str) -> IntentAnalysis:
        """Classify a user query and return analysis."""
        
        logger.info(f"Classifying query: {query[:100]}...")
        
        try:
            # First try quick pattern matching
            quick_analysis = self._quick_classify(query)
            
            # If confidence is high, return quick result
            if quick_analysis.confidence > 0.8:
                logger.info(f"Quick classification: {quick_analysis.intent} (confidence: {quick_analysis.confidence})")
                return quick_analysis
                
            # Otherwise use LLM for detailed analysis
            detailed_analysis = await self.classification_chain.ainvoke({"query": query})
            logger.info(f"LLM classification: {detailed_analysis.intent} (confidence: {detailed_analysis.confidence})")
            return detailed_analysis
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Return safe default
            return IntentAnalysis(
                intent=QueryIntent.GENERAL_CHAT,
                complexity=QueryComplexity.SIMPLE,
                confidence=0.3,
                keywords=[],
                suggested_agent="query_agent",
                requires_langgraph=False,
                data_sources=["none"],
                metadata={"error": str(e)}
            )
            
    def _quick_classify(self, query: str) -> IntentAnalysis:
        """Quick pattern-based classification for common queries."""
        
        query_lower = query.lower()
        matched_patterns = {}
        
        # Check intent patterns
        for intent, patterns in self.intent_patterns.items():
            score = 0
            matched_keywords = []
            
            # Check keywords
            for keyword in patterns["keywords"]:
                if keyword in query_lower:
                    score += 1
                    matched_keywords.append(keyword)
                    
            # Check phrases (higher weight)
            for phrase in patterns["phrases"]:
                if phrase in query_lower:
                    score += 3
                    matched_keywords.append(phrase)
                    
            if score > 0:
                matched_patterns[intent] = {
                    "score": score,
                    "keywords": matched_keywords
                }
        
        # Determine best match
        if matched_patterns:
            best_intent = max(matched_patterns.keys(), key=lambda x: matched_patterns[x]["score"])
            confidence = min(0.9, matched_patterns[best_intent]["score"] * 0.15)
            keywords = matched_patterns[best_intent]["keywords"]
        else:
            best_intent = QueryIntent.GENERAL_CHAT
            confidence = 0.3
            keywords = []
            
        # Determine complexity
        complexity = self._determine_complexity(query_lower)
        
        # Determine data sources
        data_sources = self._identify_data_sources(query_lower)
        
        return IntentAnalysis(
            intent=best_intent,
            complexity=complexity,
            confidence=confidence,
            keywords=keywords,
            suggested_agent=self._determine_suggested_agent(best_intent, complexity),
            requires_langgraph=self._requires_langgraph(best_intent, complexity),
            data_sources=data_sources,
            metadata={"method": "quick_pattern_matching"}
        )
        
    def _determine_complexity(self, query: str) -> QueryComplexity:
        """Determine query complexity based on indicators."""
        
        for complexity, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if indicator in query:
                    return complexity
                    
        # Default complexity based on query length and structure
        if len(query.split()) < 5:
            return QueryComplexity.SIMPLE
        elif "?" in query or any(word in query for word in ["how", "why", "explain"]):
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
            
    def _identify_data_sources(self, query: str) -> List[str]:
        """Identify which data sources are relevant to the query."""
        
        data_source_keywords = {
            "jira": ["jira", "ticket", "issue", "project", "sprint", "epic", "story"],
            "slack": ["slack", "message", "channel", "team", "chat", "discussion"],
            "confluence": ["confluence", "wiki", "document", "page", "documentation"],
            "github": ["github", "code", "repository", "commit", "pull request", "branch"],
            "calendar": ["calendar", "meeting", "schedule", "appointment", "event"],
            "hr_systems": ["employee", "team member", "manager", "department", "org chart"]
        }
        
        identified_sources = []
        
        for source, keywords in data_source_keywords.items():
            if any(keyword in query for keyword in keywords):
                identified_sources.append(source)
                
        return identified_sources if identified_sources else ["none"]


class QueryRouter:
    """
    Routes queries to appropriate agents based on intent analysis.
    
    Implements cascading complexity handling and intelligent agent selection.
    """
    
    def __init__(self, intent_classifier: IntentClassifier):
        self.intent_classifier = intent_classifier
        
    async def route_query(self, query: str, user_context: Dict = None) -> Dict[str, any]:
        """Route query to appropriate handler with full context."""
        
        # Analyze intent
        analysis = await self.intent_classifier.classify_query(query)
        
        # Prepare routing decision
        routing_decision = {
            "query": query,
            "analysis": analysis,
            "routing": {
                "agent": analysis.suggested_agent,
                "use_langgraph": analysis.requires_langgraph,
                "data_sources": analysis.data_sources,
                "complexity_tier": self._determine_complexity_tier(analysis.complexity)
            },
            "context": user_context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Routing decision: {analysis.intent} -> {analysis.suggested_agent} (LangGraph: {analysis.requires_langgraph})")
        
        return routing_decision
        
    def _determine_complexity_tier(self, complexity: QueryComplexity) -> int:
        """Map complexity to tier for cascading."""
        complexity_tiers = {
            QueryComplexity.SIMPLE: 1,
            QueryComplexity.MODERATE: 2, 
            QueryComplexity.COMPLEX: 3,
            QueryComplexity.EXPERT: 4
        }
        return complexity_tiers.get(complexity, 1)


# Global instances
_intent_classifier: Optional[IntentClassifier] = None
_query_router: Optional[QueryRouter] = None


def get_intent_classifier() -> IntentClassifier:
    """Get global intent classifier instance."""
    global _intent_classifier
    if _intent_classifier is None:
        from .agent_framework import BaseAgent
        # Use default LLM
        base_agent = BaseAgent.__new__(BaseAgent)
        base_agent.__init__("intent_classifier", None)
        _intent_classifier = IntentClassifier(base_agent.llm)
    return _intent_classifier


def get_query_router() -> QueryRouter:
    """Get global query router instance."""
    global _query_router
    if _query_router is None:
        _query_router = QueryRouter(get_intent_classifier())
    return _query_router