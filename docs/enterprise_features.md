# Enterprise Knowledge Management Features

Project Maestro has been extended with comprehensive enterprise knowledge management capabilities, allowing organizations to use the system not only for game development but also as a centralized knowledge hub.

## Overview

The enterprise features include:

- **Intent Analysis**: Automatically understands user queries and routes them appropriately
- **Enterprise Data Integration**: Connects to Jira, Slack, and Confluence
- **Query Agent**: Handles information retrieval with complexity-based cascading
- **RAG System**: Enhanced with enterprise document indexing and search
- **LangGraph Integration**: Orchestrates enterprise workflows alongside game development

## Architecture

```mermaid
graph TD
    A[User Query] --> B[Intent Classifier]
    B --> C{Query Intent}
    C -->|Enterprise Search| D[Query Agent]
    C -->|Game Development| E[Original Workflow]
    C -->|Complex Analysis| F[Full Orchestration]
    
    D --> G[Enterprise Data Manager]
    G --> H[Jira Connector]
    G --> I[Slack Connector] 
    G --> J[Confluence Connector]
    
    H --> K[RAG System]
    I --> K
    J --> K
    
    K --> L[Vector Store]
    L --> M[Query Response]
    
    F --> N[Multi-Agent Orchestration]
    N --> O[Complex Analysis Response]
```

## Configuration

### Environment Variables

The enterprise features are configured through environment variables or the `.env` file:

```env
# Enterprise Systems
JIRA_ENABLED=true
JIRA_BASE_URL=https://yourcompany.atlassian.net
JIRA_USERNAME=your-email@company.com
JIRA_API_TOKEN=your-jira-api-token
JIRA_PROJECT_KEYS=["PROJ", "DEV", "SUPPORT"]

SLACK_ENABLED=true
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_APP_TOKEN=xapp-your-slack-app-token
SLACK_CHANNELS=["general", "dev", "support"]
SLACK_MAX_HISTORY_DAYS=30

CONFLUENCE_ENABLED=true
CONFLUENCE_BASE_URL=https://yourcompany.atlassian.net/wiki
CONFLUENCE_USERNAME=your-email@company.com
CONFLUENCE_API_TOKEN=your-confluence-api-token
CONFLUENCE_SPACE_KEYS=["DEV", "PROJ", "KB"]

# RAG Configuration
RAG_ENABLED=true
RAG_VECTOR_STORE_TYPE=chroma  # chroma, pinecone, weaviate
RAG_EMBEDDING_MODEL=text-embedding-ada-002
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_MAX_RESULTS=10
RAG_SIMILARITY_THRESHOLD=0.7

# Vector Database (if using Chroma)
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=maestro_enterprise

# Query Agent
QUERY_AGENT_ENABLED=true
QUERY_AGENT_MODEL=gpt-4-turbo-preview
QUERY_AGENT_TEMPERATURE=0.1
QUERY_CASCADING_ENABLED=true
QUERY_COMPLEXITY_THRESHOLD=0.6

# Intent Analysis
INTENT_ANALYSIS_ENABLED=true
INTENT_ANALYSIS_MODEL=gpt-3.5-turbo
INTENT_CONFIDENCE_THRESHOLD=0.8
```

### Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install atlassian-python-api slack-sdk chromadb
   ```

2. **Configure Enterprise Systems**:
   - Create API tokens for Jira and Confluence
   - Set up a Slack app and get bot tokens
   - Configure the environment variables

3. **Initialize Vector Store**:
   ```bash
   python -m src.project_maestro.scripts.init_enterprise_data
   ```

4. **Start Data Sync**:
   ```bash
   python -m src.project_maestro.scripts.sync_enterprise_data
   ```

## Features

### Intent Classification

The system automatically analyzes incoming queries to determine:

- **Intent Type**: What kind of request this is
- **Complexity Level**: How complex the query is
- **Routing Decision**: Which agent should handle it

**Intent Types**:
- `enterprise_search`: Looking for company information
- `game_development`: Traditional game dev queries
- `project_info`: Project-specific information
- `technical_support`: Technical help requests
- `workflow_automation`: Process automation
- `complex_analysis`: Multi-step analysis needs
- `general_chat`: Conversational queries

**Complexity Levels**:
- `simple`: Direct information retrieval
- `moderate`: Requires some analysis
- `complex`: Multi-step reasoning needed
- `expert`: Requires full orchestration

### Query Agent with Cascading

The Query Agent handles enterprise queries with a 4-tier cascading system:

**Tier 1 (Simple)**: Direct information retrieval
- Quick database/API lookups
- Simple fact-finding queries
- Recent updates and notifications

**Tier 2 (Moderate)**: Basic analysis and search
- Cross-system information correlation
- Simple trend analysis
- Multi-source information compilation

**Tier 3 (Complex)**: Advanced reasoning
- Pattern recognition across data sources
- Complex query decomposition
- Multi-step analysis workflows

**Tier 4 (Expert)**: Full orchestration
- Escalates to full multi-agent system
- Complex architectural decisions
- Advanced problem-solving

### Enterprise Data Connectors

#### Jira Integration

Connects to Atlassian Jira for issue tracking data:

```python
from src.project_maestro.integrations.enterprise_connectors import JiraConnector

connector = JiraConnector(
    base_url="https://company.atlassian.net",
    username="user@company.com",
    api_token="your-token",
    project_keys=["PROJ", "DEV"]
)

# Fetch recent issues
documents = await connector.fetch_documents(days_back=30)

# Search issues
results = await connector.search_documents("authentication bug")
```

**Data Indexed**:
- Issue summaries and descriptions
- Comments and updates
- Project and component information
- Status and priority data
- User assignments and metadata

#### Slack Integration

Connects to Slack for message and conversation data:

```python
from src.project_maestro.integrations.enterprise_connectors import SlackConnector

connector = SlackConnector(
    bot_token="xoxb-your-token",
    channels=["general", "dev"],
    max_history_days=30
)

# Fetch recent messages
documents = await connector.fetch_documents()

# Search messages
results = await connector.search_documents("deployment issue")
```

**Data Indexed**:
- Channel messages and threads
- Direct message conversations
- File attachments and links
- User mentions and reactions
- Channel metadata

#### Confluence Integration

Connects to Atlassian Confluence for documentation:

```python
from src.project_maestro.integrations.enterprise_connectors import ConfluenceConnector

connector = ConfluenceConnector(
    base_url="https://company.atlassian.net/wiki",
    username="user@company.com",
    api_token="your-token",
    space_keys=["DEV", "KB"]
)

# Fetch pages
documents = await connector.fetch_documents()

# Search pages
results = await connector.search_documents("API documentation")
```

**Data Indexed**:
- Page content and titles
- Comments and attachments
- Space and label information
- Version history
- User metadata

### Unified Enterprise Search

The Enterprise Data Manager provides unified search across all connected systems:

```python
from src.project_maestro.integrations.enterprise_connectors import EnterpriseDataManager

manager = EnterpriseDataManager()

# Search across all systems
results = await manager.search_documents(
    query="authentication issues",
    max_results=20,
    systems=["jira", "slack", "confluence"]
)

# Filter by date range
recent_results = await manager.search_documents(
    query="deployment",
    date_range=(datetime.now() - timedelta(days=7), datetime.now())
)
```

## Usage Examples

### Basic Enterprise Query

```python
# User asks: "How do I create a Jira ticket?"
# System automatically:
# 1. Classifies as enterprise_search intent
# 2. Routes to Query Agent (Tier 1)
# 3. Searches Confluence for documentation
# 4. Returns relevant pages and procedures
```

### Complex Analysis Query

```python
# User asks: "Analyze the authentication issues across our systems in the last month"
# System automatically:
# 1. Classifies as complex_analysis intent
# 2. Routes to full orchestration (Tier 4)
# 3. Searches Jira for auth-related issues
# 4. Searches Slack for discussions
# 5. Analyzes patterns and trends
# 6. Provides comprehensive analysis
```

### Mixed Game Development Query

```python
# User asks: "Generate a login screen and check if there are any auth-related Jira tickets"
# System automatically:
# 1. Recognizes mixed intent
# 2. Routes to orchestrator for coordination
# 3. Uses Canvas Agent for UI generation
# 4. Uses Query Agent for Jira search
# 5. Combines results in unified response
```

## API Endpoints

### Query Processing

```http
POST /api/v1/enterprise/query
Content-Type: application/json

{
  "query": "How do I deploy the new authentication service?",
  "user_id": "user123",
  "context": {
    "project": "auth-service",
    "environment": "staging"
  }
}
```

**Response**:
```json
{
  "response": "Based on our deployment documentation and recent Slack discussions...",
  "intent": "enterprise_search",
  "complexity": "moderate",
  "sources": [
    {
      "title": "Deployment Guide",
      "url": "https://confluence.com/deployment",
      "source": "confluence",
      "relevance": 0.95
    }
  ],
  "processing_time": 1.2
}
```

### Enterprise Data Search

```http
GET /api/v1/enterprise/search?q=authentication&systems=jira,slack&limit=10
```

**Response**:
```json
{
  "results": [
    {
      "id": "PROJ-123",
      "title": "Authentication Bug Fix",
      "content": "Fixed issue with JWT token validation...",
      "source": "jira",
      "content_type": "issue",
      "created_date": "2024-01-15T10:30:00Z",
      "url": "https://jira.com/PROJ-123",
      "relevance": 0.92
    }
  ],
  "total": 25,
  "query_time": 0.15
}
```

## Monitoring and Analytics

### Query Analytics

The system tracks:
- Query volume and patterns
- Response times by complexity tier
- Source system usage
- User satisfaction scores

### Data Freshness

Enterprise data is synced according to configured intervals:
- **Jira**: Every 4 hours
- **Slack**: Every 1 hour  
- **Confluence**: Every 8 hours

### Performance Metrics

Monitor key metrics:
- Query response time by tier
- Enterprise connector health
- Vector store performance
- Cache hit rates

## Security Considerations

### Authentication
- All enterprise system connections use secure API tokens
- Tokens are stored encrypted in environment variables
- Regular token rotation recommended

### Data Privacy
- Enterprise data is processed locally
- Sensitive information can be filtered out
- Audit logs track all data access

### Access Control
- Role-based access to enterprise features
- Query history is user-scoped
- Administrative controls for system configuration

## Troubleshooting

### Common Issues

**Enterprise Connector Failures**:
```bash
# Check connector status
python -m src.project_maestro.scripts.check_enterprise_health

# Test individual connectors
python -m src.project_maestro.scripts.test_connector --system jira
```

**Vector Store Issues**:
```bash
# Rebuild vector store
python -m src.project_maestro.scripts.rebuild_vector_store

# Check indexing status
python -m src.project_maestro.scripts.index_status
```

**Query Performance Issues**:
- Check vector store performance
- Review similarity thresholds
- Monitor query complexity distribution
- Consider query result caching

### Debugging

Enable debug logging:
```env
LOG_LEVEL=DEBUG
LANGCHAIN_TRACING_V2=true
```

View detailed logs:
```bash
tail -f /tmp/maestro_logs/enterprise.log
```

## Future Roadmap

Planned enterprise enhancements:

- **Additional Connectors**: GitHub, GitLab, Notion, Teams
- **Advanced Analytics**: Query insights and recommendations  
- **Workflow Automation**: Automated responses and actions
- **Multi-tenancy**: Support for multiple organizations
- **Advanced Security**: SSO integration, advanced permissions
- **Mobile Support**: Native mobile query interface

## Support

For enterprise feature support:
- Documentation: See `/docs/` directory
- Issues: Create GitHub issues with `enterprise` label
- Configuration: Review `src/project_maestro/core/config.py`
- Testing: Run `pytest tests/test_enterprise_features.py`