"""
Enterprise Data Connectors for Project Maestro

Integrates with various third-party enterprise systems to provide comprehensive
knowledge management and search capabilities.

Supported Systems:
- Atlassian Jira (Issues, Projects, Comments)
- Slack (Messages, Channels, Files) 
- Confluence (Pages, Spaces, Comments)
- GitHub (Repositories, Issues, PRs)
- Google Workspace (Calendar, Docs, Drive)
"""

from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
import base64
from urllib.parse import urlencode

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from ..core.logging import get_logger
from ..core.config import settings

logger = get_logger(__name__)


@dataclass
class ConnectorConfig:
    """Configuration for enterprise connectors."""
    base_url: str
    api_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    webhook_url: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    timeout: int = 30  # seconds
    max_retries: int = 3
    extra_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class SearchOptions:
    """Options for enterprise data search."""
    limit: int = 50
    offset: int = 0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    content_types: List[str] = field(default_factory=list)
    spaces_or_channels: List[str] = field(default_factory=list)
    include_archived: bool = False
    sort_by: str = "relevance"  # relevance, date, popularity


class EnterpriseDocument(BaseModel):
    """Standardized document format for enterprise data."""
    id: str
    title: str
    content: str
    url: str
    source: str  # jira, slack, confluence, etc.
    content_type: str  # issue, message, page, etc.
    created_date: datetime
    updated_date: Optional[datetime] = None
    author: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseEnterpriseConnector(ABC):
    """Base class for enterprise system connectors."""
    
    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
        
    async def connect(self):
        """Initialize connection to enterprise system."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                headers=self._get_auth_headers(),
                timeout=timeout
            )
            
    async def disconnect(self):
        """Close connection to enterprise system."""
        if self.session:
            await self.session.close()
            self.session = None
            
    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        pass
        
    @abstractmethod
    async def search(self, query: str, options: SearchOptions = None) -> List[EnterpriseDocument]:
        """Search for content in the enterprise system."""
        pass
        
    @abstractmethod
    async def get_recent_updates(self, days: int = 7) -> List[EnterpriseDocument]:
        """Get recently updated content."""
        pass
        
    async def test_connection(self) -> bool:
        """Test connection to the enterprise system."""
        try:
            await self.connect()
            # Implement system-specific health check
            return await self._health_check()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
        finally:
            await self.disconnect()
            
    @abstractmethod
    async def _health_check(self) -> bool:
        """System-specific health check implementation."""
        pass


class JiraConnector(BaseEnterpriseConnector):
    """Connector for Atlassian Jira integration."""
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get Jira authentication headers."""
        headers = {"Content-Type": "application/json"}
        
        if self.config.api_token and self.config.username:
            # Use email + API token authentication
            auth_string = f"{self.config.username}:{self.config.api_token}"
            encoded_auth = base64.b64encode(auth_string.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_auth}"
        elif self.config.api_token:
            # Use bearer token
            headers["Authorization"] = f"Bearer {self.config.api_token}"
            
        headers.update(self.config.extra_headers)
        return headers
        
    async def search(self, query: str, options: SearchOptions = None) -> List[EnterpriseDocument]:
        """Search Jira issues and comments."""
        options = options or SearchOptions()
        
        # Build JQL query
        jql_parts = [f'text ~ "{query}"']
        
        if options.start_date:
            jql_parts.append(f'updated >= "{options.start_date.strftime("%Y-%m-%d")}"')
        if options.end_date:
            jql_parts.append(f'updated <= "{options.end_date.strftime("%Y-%m-%d")}"')
        if options.spaces_or_channels:  # Projects in Jira
            projects = ", ".join(f'"{p}"' for p in options.spaces_or_channels)
            jql_parts.append(f"project in ({projects})")
            
        jql = " AND ".join(jql_parts)
        
        params = {
            "jql": jql,
            "maxResults": options.limit,
            "startAt": options.offset,
            "fields": "summary,description,created,updated,reporter,assignee,status,priority,issuetype,project,comment",
            "expand": "changelog"
        }
        
        try:
            url = f"{self.config.base_url}/rest/api/3/search"
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._process_jira_issues(data.get("issues", []))
                else:
                    logger.error(f"Jira search failed: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Jira search error: {e}")
            return []
            
    async def _process_jira_issues(self, issues: List[Dict]) -> List[EnterpriseDocument]:
        """Process Jira issues into standardized documents."""
        documents = []
        
        for issue in issues:
            fields = issue.get("fields", {})
            key = issue.get("key", "")
            
            # Main issue document
            content_parts = []
            if fields.get("summary"):
                content_parts.append(f"Summary: {fields['summary']}")
            if fields.get("description"):
                content_parts.append(f"Description: {fields['description']}")
                
            doc = EnterpriseDocument(
                id=f"jira-{key}",
                title=fields.get("summary", f"Issue {key}"),
                content="\n\n".join(content_parts),
                url=f"{self.config.base_url}/browse/{key}",
                source="jira",
                content_type="issue",
                created_date=self._parse_jira_date(fields.get("created")),
                updated_date=self._parse_jira_date(fields.get("updated")),
                author=fields.get("reporter", {}).get("displayName", "Unknown"),
                metadata={
                    "key": key,
                    "status": fields.get("status", {}).get("name"),
                    "priority": fields.get("priority", {}).get("name"),
                    "assignee": fields.get("assignee", {}).get("displayName"),
                    "project": fields.get("project", {}).get("name"),
                    "issue_type": fields.get("issuetype", {}).get("name")
                }
            )
            documents.append(doc)
            
            # Add comments as separate documents
            comments = fields.get("comment", {}).get("comments", [])
            for comment in comments:
                if comment.get("body"):
                    comment_doc = EnterpriseDocument(
                        id=f"jira-{key}-comment-{comment.get('id')}",
                        title=f"Comment on {key}",
                        content=comment["body"],
                        url=f"{self.config.base_url}/browse/{key}?focusedCommentId={comment.get('id')}",
                        source="jira",
                        content_type="comment",
                        created_date=self._parse_jira_date(comment.get("created")),
                        updated_date=self._parse_jira_date(comment.get("updated")),
                        author=comment.get("author", {}).get("displayName", "Unknown"),
                        metadata={
                            "parent_issue": key,
                            "comment_id": comment.get("id")
                        }
                    )
                    documents.append(comment_doc)
                    
        return documents
        
    def _parse_jira_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse Jira date format."""
        if not date_str:
            return None
        try:
            # Jira uses ISO format: 2023-08-15T10:30:45.123+0000
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except:
            return None
            
    async def get_recent_updates(self, days: int = 7) -> List[EnterpriseDocument]:
        """Get recently updated Jira issues."""
        cutoff_date = datetime.now() - timedelta(days=days)
        options = SearchOptions(start_date=cutoff_date, limit=100)
        return await self.search("*", options)
        
    async def _health_check(self) -> bool:
        """Check Jira connectivity."""
        try:
            url = f"{self.config.base_url}/rest/api/3/myself"
            async with self.session.get(url) as response:
                return response.status == 200
        except:
            return False


class SlackConnector(BaseEnterpriseConnector):
    """Connector for Slack integration."""
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get Slack authentication headers."""
        headers = {
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json"
        }
        headers.update(self.config.extra_headers)
        return headers
        
    async def search(self, query: str, options: SearchOptions = None) -> List[EnterpriseDocument]:
        """Search Slack messages and files."""
        options = options or SearchOptions()
        
        # Build Slack search query
        search_query = query
        if options.spaces_or_channels:
            channel_filter = " ".join(f"in:#{channel}" for channel in options.spaces_or_channels)
            search_query = f"{query} {channel_filter}"
            
        params = {
            "query": search_query,
            "count": options.limit,
            "page": options.offset // options.limit + 1,
            "sort": "timestamp" if options.sort_by == "date" else "score"
        }
        
        if options.start_date:
            params["query"] += f" after:{int(options.start_date.timestamp())}"
        if options.end_date:
            params["query"] += f" before:{int(options.end_date.timestamp())}"
            
        try:
            url = f"{self.config.base_url}/api/search.messages"
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok"):
                        return await self._process_slack_messages(data.get("messages", {}).get("matches", []))
                    else:
                        logger.error(f"Slack search failed: {data.get('error')}")
                else:
                    logger.error(f"Slack API error: {response.status}")
            return []
            
        except Exception as e:
            logger.error(f"Slack search error: {e}")
            return []
            
    async def _process_slack_messages(self, messages: List[Dict]) -> List[EnterpriseDocument]:
        """Process Slack messages into standardized documents."""
        documents = []
        
        for message in messages:
            text = message.get("text", "")
            ts = message.get("ts", "")
            channel = message.get("channel", {})
            user = message.get("user", "")
            
            # Get user info
            user_name = await self._get_user_name(user) if user else "Unknown"
            
            doc = EnterpriseDocument(
                id=f"slack-{channel.get('id', '')}-{ts}",
                title=f"Message in #{channel.get('name', 'unknown')}",
                content=text,
                url=f"{self.config.base_url}/archives/{channel.get('id', '')}/p{ts.replace('.', '')}",
                source="slack",
                content_type="message",
                created_date=datetime.fromtimestamp(float(ts)) if ts else datetime.now(),
                updated_date=None,
                author=user_name,
                metadata={
                    "channel_id": channel.get("id"),
                    "channel_name": channel.get("name"),
                    "message_type": message.get("type", "message"),
                    "thread_ts": message.get("thread_ts")
                }
            )
            documents.append(doc)
            
        return documents
        
    async def _get_user_name(self, user_id: str) -> str:
        """Get user display name from Slack."""
        try:
            url = f"{self.config.base_url}/api/users.info"
            params = {"user": user_id}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok"):
                        user = data.get("user", {})
                        return user.get("real_name") or user.get("name", user_id)
            return user_id
        except:
            return user_id
            
    async def get_recent_updates(self, days: int = 7) -> List[EnterpriseDocument]:
        """Get recent Slack messages."""
        cutoff_date = datetime.now() - timedelta(days=days)
        options = SearchOptions(start_date=cutoff_date, limit=100)
        return await self.search("*", options)
        
    async def _health_check(self) -> bool:
        """Check Slack connectivity."""
        try:
            url = f"{self.config.base_url}/api/auth.test"
            async with self.session.post(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("ok", False)
            return False
        except:
            return False


class ConfluenceConnector(BaseEnterpriseConnector):
    """Connector for Atlassian Confluence integration."""
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get Confluence authentication headers."""
        headers = {"Content-Type": "application/json"}
        
        if self.config.api_token and self.config.username:
            auth_string = f"{self.config.username}:{self.config.api_token}"
            encoded_auth = base64.b64encode(auth_string.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_auth}"
        elif self.config.api_token:
            headers["Authorization"] = f"Bearer {self.config.api_token}"
            
        headers.update(self.config.extra_headers)
        return headers
        
    async def search(self, query: str, options: SearchOptions = None) -> List[EnterpriseDocument]:
        """Search Confluence pages and comments."""
        options = options or SearchOptions()
        
        # Build CQL (Confluence Query Language) query
        cql_parts = [f'text ~ "{query}"']
        
        if options.content_types:
            types = ", ".join(f'"{t}"' for t in options.content_types)
            cql_parts.append(f"type in ({types})")
        else:
            cql_parts.append('type in ("page", "blogpost")')
            
        if options.spaces_or_channels:
            spaces = ", ".join(f'"{s}"' for s in options.spaces_or_channels)
            cql_parts.append(f"space in ({spaces})")
            
        if options.start_date:
            cql_parts.append(f'lastModified >= "{options.start_date.strftime("%Y-%m-%d")}"')
        if options.end_date:
            cql_parts.append(f'lastModified <= "{options.end_date.strftime("%Y-%m-%d")}"')
            
        cql = " AND ".join(cql_parts)
        
        params = {
            "cql": cql,
            "limit": options.limit,
            "start": options.offset,
            "expand": "version,space,body.view,metadata.labels"
        }
        
        try:
            url = f"{self.config.base_url}/wiki/rest/api/content/search"
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._process_confluence_content(data.get("results", []))
                else:
                    logger.error(f"Confluence search failed: {response.status}")
            return []
            
        except Exception as e:
            logger.error(f"Confluence search error: {e}")
            return []
            
    async def _process_confluence_content(self, content_items: List[Dict]) -> List[EnterpriseDocument]:
        """Process Confluence content into standardized documents."""
        documents = []
        
        for item in content_items:
            content_id = item.get("id", "")
            title = item.get("title", "")
            content_type = item.get("type", "page")
            
            # Extract content
            body = item.get("body", {}).get("view", {}).get("value", "")
            
            # Clean HTML content (basic cleaning)
            import re
            clean_content = re.sub(r'<[^>]+>', '', body)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            
            version = item.get("version", {})
            space = item.get("space", {})
            
            doc = EnterpriseDocument(
                id=f"confluence-{content_id}",
                title=title,
                content=clean_content,
                url=f"{self.config.base_url}/wiki{item.get('_links', {}).get('webui', '')}",
                source="confluence",
                content_type=content_type,
                created_date=self._parse_confluence_date(version.get("when")),
                updated_date=self._parse_confluence_date(version.get("when")),
                author=version.get("by", {}).get("displayName", "Unknown"),
                metadata={
                    "content_id": content_id,
                    "space_key": space.get("key"),
                    "space_name": space.get("name"),
                    "version": version.get("number"),
                    "content_type": content_type,
                    "labels": [label.get("name") for label in item.get("metadata", {}).get("labels", {}).get("results", [])]
                }
            )
            documents.append(doc)
            
        return documents
        
    def _parse_confluence_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse Confluence date format."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except:
            return None
            
    async def get_recent_updates(self, days: int = 7) -> List[EnterpriseDocument]:
        """Get recently updated Confluence content."""
        cutoff_date = datetime.now() - timedelta(days=days)
        options = SearchOptions(start_date=cutoff_date, limit=100)
        return await self.search("*", options)
        
    async def _health_check(self) -> bool:
        """Check Confluence connectivity."""
        try:
            url = f"{self.config.base_url}/wiki/rest/api/space"
            params = {"limit": 1}
            async with self.session.get(url, params=params) as response:
                return response.status == 200
        except:
            return False


class EnterpriseDataManager:
    """
    Manages multiple enterprise connectors and provides unified search interface.
    """
    
    def __init__(self):
        self.connectors: Dict[str, BaseEnterpriseConnector] = {}
        self.connector_configs: Dict[str, ConnectorConfig] = {}
        
    def register_connector(self, name: str, connector_class: type, config: ConnectorConfig):
        """Register an enterprise connector."""
        self.connector_configs[name] = config
        self.connectors[name] = connector_class(config)
        logger.info(f"Registered {name} connector")
        
    async def search_all(self, query: str, options: SearchOptions = None) -> Dict[str, List[EnterpriseDocument]]:
        """Search across all registered connectors."""
        results = {}
        
        # Search all connectors concurrently
        tasks = []
        for name, connector in self.connectors.items():
            task = self._search_connector(name, connector, query, options)
            tasks.append(task)
            
        connector_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (name, _) in enumerate(self.connectors.items()):
            result = connector_results[i]
            if isinstance(result, Exception):
                logger.error(f"Error searching {name}: {result}")
                results[name] = []
            else:
                results[name] = result
                
        return results
        
    async def _search_connector(
        self, 
        name: str, 
        connector: BaseEnterpriseConnector, 
        query: str, 
        options: SearchOptions
    ) -> List[EnterpriseDocument]:
        """Search a single connector with error handling."""
        try:
            async with connector:
                return await connector.search(query, options)
        except Exception as e:
            logger.error(f"Connector {name} search failed: {e}")
            return []
            
    async def get_unified_results(self, query: str, options: SearchOptions = None) -> List[EnterpriseDocument]:
        """Get unified search results from all connectors."""
        all_results = await self.search_all(query, options)
        
        # Combine and sort results
        unified_results = []
        for connector_results in all_results.values():
            unified_results.extend(connector_results)
            
        # Sort by relevance/date
        if options and options.sort_by == "date":
            unified_results.sort(key=lambda x: x.created_date or datetime.min, reverse=True)
        else:
            # For relevance, we could implement more sophisticated scoring
            unified_results.sort(key=lambda x: x.title.lower())
            
        return unified_results[:options.limit if options else 50]
        
    async def test_all_connections(self) -> Dict[str, bool]:
        """Test connectivity to all registered systems."""
        results = {}
        for name, connector in self.connectors.items():
            results[name] = await connector.test_connection()
        return results


# Global enterprise data manager instance
_enterprise_manager: Optional[EnterpriseDataManager] = None


def get_enterprise_manager() -> EnterpriseDataManager:
    """Get global enterprise data manager."""
    global _enterprise_manager
    if _enterprise_manager is None:
        _enterprise_manager = EnterpriseDataManager()
        
        # Auto-register connectors based on settings
        if hasattr(settings, 'jira_base_url') and settings.jira_base_url:
            jira_config = ConnectorConfig(
                base_url=settings.jira_base_url,
                username=getattr(settings, 'jira_username', None),
                api_token=getattr(settings, 'jira_api_token', None)
            )
            _enterprise_manager.register_connector("jira", JiraConnector, jira_config)
            
        if hasattr(settings, 'slack_base_url') and settings.slack_base_url:
            slack_config = ConnectorConfig(
                base_url=settings.slack_base_url,
                api_token=getattr(settings, 'slack_api_token', None)
            )
            _enterprise_manager.register_connector("slack", SlackConnector, slack_config)
            
        if hasattr(settings, 'confluence_base_url') and settings.confluence_base_url:
            confluence_config = ConnectorConfig(
                base_url=settings.confluence_base_url,
                username=getattr(settings, 'confluence_username', None),
                api_token=getattr(settings, 'confluence_api_token', None)
            )
            _enterprise_manager.register_connector("confluence", ConfluenceConnector, confluence_config)
            
    return _enterprise_manager


async def index_enterprise_data():
    """Index data from all enterprise systems into vector store."""
    manager = get_enterprise_manager()
    
    # Get recent updates from all systems
    all_documents = []
    
    for name, connector in manager.connectors.items():
        try:
            async with connector:
                recent_docs = await connector.get_recent_updates(days=30)
                all_documents.extend(recent_docs)
                logger.info(f"Retrieved {len(recent_docs)} documents from {name}")
        except Exception as e:
            logger.error(f"Failed to index data from {name}: {e}")
            
    # Convert to LangChain documents for RAG system
    langchain_docs = []
    for doc in all_documents:
        langchain_doc = Document(
            page_content=f"Title: {doc.title}\n\nContent: {doc.content}",
            metadata={
                "source": doc.source,
                "url": doc.url,
                "title": doc.title,
                "author": doc.author,
                "created_date": doc.created_date.isoformat() if doc.created_date else None,
                "content_type": doc.content_type,
                **doc.metadata
            }
        )
        langchain_docs.append(langchain_doc)
        
    logger.info(f"Prepared {len(langchain_docs)} documents for indexing")
    return langchain_docs