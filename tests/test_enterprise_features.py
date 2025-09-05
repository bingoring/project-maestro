"""Tests for enterprise knowledge management features."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.project_maestro.core.intent_classifier import (
    IntentClassifier, QueryRouter, QueryIntent, QueryComplexity
)
from src.project_maestro.integrations.enterprise_connectors import (
    JiraConnector, SlackConnector, ConfluenceConnector, 
    EnterpriseDataManager, EnterpriseDocument
)
from src.project_maestro.agents.query_agent import QueryAgent
from src.project_maestro.core.config import get_enterprise_config


class TestIntentClassifier:
    """Test intent classification and query routing."""
    
    @pytest.fixture
    def intent_classifier(self):
        return IntentClassifier()
    
    @pytest.fixture
    def query_router(self):
        return QueryRouter()
    
    @pytest.mark.asyncio
    async def test_classify_enterprise_intent(self, intent_classifier):
        """Test enterprise query intent classification."""
        query = "How do I create a Jira ticket for this bug?"
        
        with patch.object(intent_classifier, 'llm') as mock_llm:
            mock_llm.invoke.return_value.content = '''
            {
                "intent": "enterprise_search",
                "complexity": "simple",
                "confidence": 0.9,
                "reasoning": "User asking about Jira ticket creation process"
            }
            '''
            
            result = await intent_classifier.classify_intent(query)
            
            assert result.intent == QueryIntent.ENTERPRISE_SEARCH
            assert result.complexity == QueryComplexity.SIMPLE
            assert result.confidence == 0.9
            assert "Jira" in result.reasoning
    
    @pytest.mark.asyncio
    async def test_classify_complex_technical_intent(self, intent_classifier):
        """Test complex technical query classification."""
        query = "Analyze the performance bottlenecks in our microservices architecture and suggest optimization strategies"
        
        with patch.object(intent_classifier, 'llm') as mock_llm:
            mock_llm.invoke.return_value.content = '''
            {
                "intent": "complex_analysis",
                "complexity": "expert",
                "confidence": 0.95,
                "reasoning": "Complex architectural analysis requiring expert knowledge"
            }
            '''
            
            result = await intent_classifier.classify_intent(query)
            
            assert result.intent == QueryIntent.COMPLEX_ANALYSIS
            assert result.complexity == QueryComplexity.EXPERT
            assert result.confidence == 0.95
    
    def test_query_routing_simple_enterprise(self, query_router):
        """Test routing for simple enterprise queries."""
        mock_analysis = Mock()
        mock_analysis.intent = QueryIntent.ENTERPRISE_SEARCH
        mock_analysis.complexity = QueryComplexity.SIMPLE
        
        decision = query_router.route_query(mock_analysis)
        
        assert decision.agent == "query_agent"
        assert decision.cascading_tier == 1
        assert decision.use_enterprise_connectors is True
    
    def test_query_routing_complex_analysis(self, query_router):
        """Test routing for complex analysis queries."""
        mock_analysis = Mock()
        mock_analysis.intent = QueryIntent.COMPLEX_ANALYSIS
        mock_analysis.complexity = QueryComplexity.EXPERT
        
        decision = query_router.route_query(mock_analysis)
        
        assert decision.agent == "orchestrator"
        assert decision.cascading_tier == 4
        assert decision.requires_full_orchestration is True


class TestEnterpriseConnectors:
    """Test enterprise system connectors."""
    
    @pytest.fixture
    def jira_config(self):
        return {
            "base_url": "https://company.atlassian.net",
            "username": "test@company.com",
            "api_token": "test_token",
            "project_keys": ["PROJ"]
        }
    
    @pytest.fixture
    def slack_config(self):
        return {
            "bot_token": "xoxb-test-token",
            "channels": ["general", "dev"],
            "max_history_days": 30
        }
    
    @pytest.fixture
    def confluence_config(self):
        return {
            "base_url": "https://company.atlassian.net/wiki",
            "username": "test@company.com", 
            "api_token": "test_token",
            "space_keys": ["DEV", "PROJ"]
        }
    
    @pytest.mark.asyncio
    async def test_jira_connector_fetch_issues(self, jira_config):
        """Test Jira connector fetching issues."""
        connector = JiraConnector(**jira_config)
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock Jira API response
            mock_response = Mock()
            mock_response.json.return_value = {
                "issues": [{
                    "id": "PROJ-123",
                    "key": "PROJ-123",
                    "fields": {
                        "summary": "Test Issue",
                        "description": "Test Description",
                        "created": "2024-01-01T00:00:00.000+0000",
                        "updated": "2024-01-01T00:00:00.000+0000",
                        "reporter": {"displayName": "Test User"}
                    },
                    "self": "https://company.atlassian.net/rest/api/2/issue/PROJ-123"
                }]
            }
            mock_get.return_value.__aenter__.return_value = mock_response
            
            documents = await connector.fetch_documents()
            
            assert len(documents) == 1
            assert documents[0].id == "PROJ-123"
            assert documents[0].title == "Test Issue"
            assert documents[0].source == "jira"
            assert documents[0].content_type == "issue"
    
    @pytest.mark.asyncio
    async def test_slack_connector_fetch_messages(self, slack_config):
        """Test Slack connector fetching messages."""
        connector = SlackConnector(**slack_config)
        
        with patch('slack_sdk.WebClient.conversations_history') as mock_history:
            mock_history.return_value = {
                "messages": [{
                    "ts": "1640995200.000000",
                    "text": "Test message",
                    "user": "U123456",
                    "channel": "C123456"
                }]
            }
            
            with patch('slack_sdk.WebClient.users_info') as mock_user_info:
                mock_user_info.return_value = {
                    "user": {"real_name": "Test User"}
                }
                
                documents = await connector.fetch_documents()
                
                assert len(documents) > 0
                assert documents[0].content == "Test message"
                assert documents[0].source == "slack"
                assert documents[0].content_type == "message"
                assert documents[0].author == "Test User"
    
    @pytest.mark.asyncio
    async def test_enterprise_data_manager_search(self):
        """Test enterprise data manager unified search."""
        manager = EnterpriseDataManager()
        
        # Mock connectors
        mock_jira = Mock()
        mock_jira.search_documents.return_value = [
            EnterpriseDocument(
                id="PROJ-123",
                title="Test Issue",
                content="Bug in authentication",
                url="https://jira.com/PROJ-123",
                source="jira",
                content_type="issue",
                created_date=datetime.now(),
                author="Test User"
            )
        ]
        
        mock_slack = Mock()
        mock_slack.search_documents.return_value = [
            EnterpriseDocument(
                id="slack-123",
                title="Channel Message",
                content="Authentication fix deployed",
                url="https://slack.com/message/123",
                source="slack",
                content_type="message",
                created_date=datetime.now(),
                author="Dev User"
            )
        ]
        
        manager.connectors = {"jira": mock_jira, "slack": mock_slack}
        
        results = await manager.search_documents("authentication")
        
        assert len(results) == 2
        assert any(doc.source == "jira" for doc in results)
        assert any(doc.source == "slack" for doc in results)


class TestQueryAgent:
    """Test query agent with cascading."""
    
    @pytest.fixture
    def query_agent(self):
        return QueryAgent()
    
    @pytest.mark.asyncio
    async def test_simple_query_cascading(self, query_agent):
        """Test simple query handling (tier 1)."""
        with patch.object(query_agent, '_handle_simple_query') as mock_handler:
            mock_handler.return_value = "Simple response"
            
            result = await query_agent.process_query(
                "What is our Jira URL?", 
                complexity_tier=1
            )
            
            assert result == "Simple response"
            mock_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_complex_query_escalation(self, query_agent):
        """Test query escalation for complex queries."""
        complex_query = "Analyze cross-team dependencies and suggest optimization"
        
        with patch.object(query_agent, '_handle_expert_query') as mock_handler:
            mock_handler.return_value = "Expert analysis response"
            
            result = await query_agent.process_query(complex_query, complexity_tier=4)
            
            assert result == "Expert analysis response"
            mock_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enterprise_search_integration(self, query_agent):
        """Test integration with enterprise search."""
        with patch.object(query_agent, 'enterprise_manager') as mock_manager:
            mock_manager.search_documents.return_value = [
                EnterpriseDocument(
                    id="test-1",
                    title="Test Document",
                    content="Relevant information",
                    url="https://example.com",
                    source="confluence",
                    content_type="page",
                    created_date=datetime.now(),
                    author="Test Author"
                )
            ]
            
            result = await query_agent._search_enterprise_data("test query")
            
            assert len(result) == 1
            assert result[0].title == "Test Document"
            mock_manager.search_documents.assert_called_once_with("test query")


class TestEnterpriseConfiguration:
    """Test enterprise configuration validation."""
    
    def test_enterprise_config_access(self):
        """Test enterprise configuration access."""
        config = get_enterprise_config()
        
        assert "jira" in config
        assert "slack" in config
        assert "confluence" in config
        assert "rag" in config
        assert "query_agent" in config
        
        # Check structure
        assert "enabled" in config["jira"]
        assert "base_url" in config["jira"]
        assert "vector_store_type" in config["rag"]
        assert "cascading_enabled" in config["query_agent"]
    
    @patch('src.project_maestro.core.config.settings')
    def test_enterprise_validation_errors(self, mock_settings):
        """Test enterprise configuration validation."""
        from src.project_maestro.core.config import validate_enterprise_services
        
        # Mock settings with Jira enabled but missing config
        mock_settings.jira_enabled = True
        mock_settings.jira_base_url = None
        mock_settings.jira_username = None
        mock_settings.jira_api_token = None
        mock_settings.is_production.return_value = True
        
        with pytest.raises(ValueError) as exc_info:
            validate_enterprise_services()
        
        assert "JIRA_BASE_URL is required" in str(exc_info.value)
        assert "JIRA_USERNAME is required" in str(exc_info.value)
        assert "JIRA_API_TOKEN is required" in str(exc_info.value)


@pytest.mark.integration
class TestEnterpriseIntegration:
    """Integration tests for enterprise features."""
    
    @pytest.mark.asyncio
    async def test_full_enterprise_workflow(self):
        """Test complete enterprise query workflow."""
        # This would be a full end-to-end test
        # Mock LangGraph orchestrator and test the full flow
        
        query = "How do I resolve authentication issues in our system?"
        
        # 1. Intent classification
        # 2. Query routing
        # 3. Enterprise data search
        # 4. Query agent processing
        # 5. Response generation
        
        # This is a placeholder for the full workflow test
        assert True  # Replace with actual integration test
    
    @pytest.mark.asyncio
    async def test_enterprise_data_sync(self):
        """Test enterprise data synchronization."""
        # Test periodic sync of data from enterprise systems
        # This would test the background jobs and data freshness
        
        assert True  # Replace with actual sync test