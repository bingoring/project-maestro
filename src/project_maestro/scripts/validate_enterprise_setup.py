#!/usr/bin/env python3
"""
Enterprise setup validation script.
Validates enterprise system configurations and connectivity.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import aiohttp
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.project_maestro.core.config import settings, get_enterprise_config, validate_enterprise_services
from src.project_maestro.integrations.enterprise_connectors import (
    JiraConnector, SlackConnector, ConfluenceConnector
)


class EnterpriseValidator:
    """Validates enterprise system setup and connectivity."""
    
    def __init__(self):
        self.config = get_enterprise_config()
        self.results: List[Dict] = []
    
    def log_result(self, system: str, test: str, status: str, message: str):
        """Log a test result."""
        self.results.append({
            "system": system,
            "test": test,
            "status": status,  # "PASS", "FAIL", "WARN", "SKIP"
            "message": message,
            "timestamp": datetime.now()
        })
        
        # Color coding for console output
        colors = {
            "PASS": "\033[92m",  # Green
            "FAIL": "\033[91m",  # Red
            "WARN": "\033[93m",  # Yellow
            "SKIP": "\033[94m",  # Blue
        }
        reset = "\033[0m"
        
        color = colors.get(status, "")
        print(f"{color}[{status}]{reset} {system} - {test}: {message}")
    
    def validate_basic_config(self):
        """Validate basic application configuration."""
        print("\n=== Basic Configuration ===")
        
        # Check required AI API keys
        if settings.openai_api_key:
            self.log_result("Core", "OpenAI API Key", "PASS", "OpenAI API key is configured")
        else:
            self.log_result("Core", "OpenAI API Key", "FAIL", "OpenAI API key is missing")
        
        if settings.anthropic_api_key:
            self.log_result("Core", "Anthropic API Key", "PASS", "Anthropic API key is configured")
        else:
            self.log_result("Core", "Anthropic API Key", "WARN", "Anthropic API key is missing (optional)")
        
        # Check directories
        try:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            self.log_result("Core", "Data Directory", "PASS", f"Data directory exists: {settings.data_dir}")
        except Exception as e:
            self.log_result("Core", "Data Directory", "FAIL", f"Cannot create data directory: {e}")
        
        try:
            settings.logs_dir.mkdir(parents=True, exist_ok=True)
            self.log_result("Core", "Logs Directory", "PASS", f"Logs directory exists: {settings.logs_dir}")
        except Exception as e:
            self.log_result("Core", "Logs Directory", "FAIL", f"Cannot create logs directory: {e}")
    
    def validate_enterprise_config(self):
        """Validate enterprise system configurations."""
        print("\n=== Enterprise Configuration ===")
        
        try:
            validate_enterprise_services()
            self.log_result("Enterprise", "Configuration Validation", "PASS", "All enterprise configurations are valid")
        except ValueError as e:
            self.log_result("Enterprise", "Configuration Validation", "FAIL", str(e))
        except Exception as e:
            self.log_result("Enterprise", "Configuration Validation", "FAIL", f"Unexpected error: {e}")
    
    async def validate_jira_connectivity(self):
        """Validate Jira connectivity."""
        print("\n=== Jira Connectivity ===")
        
        if not self.config["jira"]["enabled"]:
            self.log_result("Jira", "Connectivity", "SKIP", "Jira integration is disabled")
            return
        
        try:
            connector = JiraConnector(
                base_url=self.config["jira"]["base_url"],
                username=self.config["jira"]["username"],
                api_token=self.config["jira"]["api_token"],
                project_keys=self.config["jira"]["project_keys"]
            )
            
            # Test basic connectivity
            async with aiohttp.ClientSession() as session:
                # Test authentication
                auth_url = f"{self.config['jira']['base_url']}/rest/api/2/myself"
                async with session.get(auth_url, auth=connector.auth) as response:
                    if response.status == 200:
                        user_info = await response.json()
                        self.log_result("Jira", "Authentication", "PASS", 
                                      f"Connected as {user_info.get('displayName', 'Unknown')}")
                    else:
                        self.log_result("Jira", "Authentication", "FAIL", 
                                      f"Authentication failed: {response.status}")
                        return
                
                # Test project access
                if self.config["jira"]["project_keys"]:
                    project_key = self.config["jira"]["project_keys"][0]
                    project_url = f"{self.config['jira']['base_url']}/rest/api/2/project/{project_key}"
                    async with session.get(project_url, auth=connector.auth) as response:
                        if response.status == 200:
                            project_info = await response.json()
                            self.log_result("Jira", "Project Access", "PASS", 
                                          f"Can access project: {project_info.get('name', project_key)}")
                        else:
                            self.log_result("Jira", "Project Access", "WARN", 
                                          f"Cannot access project {project_key}: {response.status}")
        
        except Exception as e:
            self.log_result("Jira", "Connectivity", "FAIL", f"Connection error: {e}")
    
    async def validate_slack_connectivity(self):
        """Validate Slack connectivity."""
        print("\n=== Slack Connectivity ===")
        
        if not self.config["slack"]["enabled"]:
            self.log_result("Slack", "Connectivity", "SKIP", "Slack integration is disabled")
            return
        
        try:
            from slack_sdk import WebClient
            from slack_sdk.errors import SlackApiError
            
            client = WebClient(token=self.config["slack"]["bot_token"])
            
            # Test authentication
            try:
                response = client.auth_test()
                if response["ok"]:
                    self.log_result("Slack", "Authentication", "PASS", 
                                  f"Connected as {response['user']} in {response['team']}")
                else:
                    self.log_result("Slack", "Authentication", "FAIL", "Authentication failed")
                    return
            except SlackApiError as e:
                self.log_result("Slack", "Authentication", "FAIL", f"API error: {e}")
                return
            
            # Test channel access
            if self.config["slack"]["channels"]:
                try:
                    channels_response = client.conversations_list(types="public_channel,private_channel")
                    if channels_response["ok"]:
                        available_channels = {ch["name"] for ch in channels_response["channels"]}
                        configured_channels = set(self.config["slack"]["channels"])
                        
                        accessible = configured_channels.intersection(available_channels)
                        inaccessible = configured_channels - available_channels
                        
                        if accessible:
                            self.log_result("Slack", "Channel Access", "PASS", 
                                          f"Can access channels: {', '.join(accessible)}")
                        if inaccessible:
                            self.log_result("Slack", "Channel Access", "WARN", 
                                          f"Cannot access channels: {', '.join(inaccessible)}")
                except SlackApiError as e:
                    self.log_result("Slack", "Channel Access", "WARN", f"Cannot list channels: {e}")
        
        except ImportError:
            self.log_result("Slack", "Dependencies", "FAIL", "slack-sdk not installed")
        except Exception as e:
            self.log_result("Slack", "Connectivity", "FAIL", f"Connection error: {e}")
    
    async def validate_confluence_connectivity(self):
        """Validate Confluence connectivity."""
        print("\n=== Confluence Connectivity ===")
        
        if not self.config["confluence"]["enabled"]:
            self.log_result("Confluence", "Connectivity", "SKIP", "Confluence integration is disabled")
            return
        
        try:
            from atlassian import Confluence
            
            confluence = Confluence(
                url=self.config["confluence"]["base_url"],
                username=self.config["confluence"]["username"],
                password=self.config["confluence"]["api_token"],
                cloud=True
            )
            
            # Test connectivity by getting user info
            try:
                user_info = confluence.get_current_user()
                self.log_result("Confluence", "Authentication", "PASS", 
                              f"Connected as {user_info.get('displayName', 'Unknown')}")
            except Exception as e:
                self.log_result("Confluence", "Authentication", "FAIL", f"Authentication failed: {e}")
                return
            
            # Test space access
            if self.config["confluence"]["space_keys"]:
                try:
                    spaces = confluence.get_all_spaces(limit=100)
                    available_spaces = {space["key"] for space in spaces["results"]}
                    configured_spaces = set(self.config["confluence"]["space_keys"])
                    
                    accessible = configured_spaces.intersection(available_spaces)
                    inaccessible = configured_spaces - available_spaces
                    
                    if accessible:
                        self.log_result("Confluence", "Space Access", "PASS", 
                                      f"Can access spaces: {', '.join(accessible)}")
                    if inaccessible:
                        self.log_result("Confluence", "Space Access", "WARN", 
                                      f"Cannot access spaces: {', '.join(inaccessible)}")
                except Exception as e:
                    self.log_result("Confluence", "Space Access", "WARN", f"Cannot list spaces: {e}")
        
        except ImportError:
            self.log_result("Confluence", "Dependencies", "FAIL", "atlassian-python-api not installed")
        except Exception as e:
            self.log_result("Confluence", "Connectivity", "FAIL", f"Connection error: {e}")
    
    def validate_vector_store(self):
        """Validate vector store configuration."""
        print("\n=== Vector Store Configuration ===")
        
        if not self.config["rag"]["enabled"]:
            self.log_result("RAG", "Vector Store", "SKIP", "RAG system is disabled")
            return
        
        store_type = self.config["rag"]["vector_store_type"]
        
        if store_type == "chroma":
            try:
                import chromadb
                self.log_result("RAG", "Chroma Dependencies", "PASS", "ChromaDB is installed")
                
                # Test Chroma connection
                try:
                    client = chromadb.HttpClient(
                        host=settings.chroma_host, 
                        port=settings.chroma_port
                    )
                    collections = client.list_collections()
                    self.log_result("RAG", "Chroma Connectivity", "PASS", 
                                  f"Connected to ChromaDB at {settings.chroma_host}:{settings.chroma_port}")
                except Exception as e:
                    self.log_result("RAG", "Chroma Connectivity", "FAIL", 
                                  f"Cannot connect to ChromaDB: {e}")
            except ImportError:
                self.log_result("RAG", "Chroma Dependencies", "FAIL", "chromadb not installed")
        
        elif store_type == "pinecone":
            if settings.pinecone_api_key:
                self.log_result("RAG", "Pinecone Config", "PASS", "Pinecone API key is configured")
                # Could add actual Pinecone connectivity test here
            else:
                self.log_result("RAG", "Pinecone Config", "FAIL", "Pinecone API key is missing")
        
        else:
            self.log_result("RAG", "Vector Store", "WARN", f"Unknown vector store type: {store_type}")
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        # Count results by status
        status_counts = {}
        for result in self.results:
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Print counts
        for status in ["PASS", "WARN", "FAIL", "SKIP"]:
            count = status_counts.get(status, 0)
            if count > 0:
                colors = {
                    "PASS": "\033[92m",  # Green
                    "FAIL": "\033[91m",  # Red
                    "WARN": "\033[93m",  # Yellow
                    "SKIP": "\033[94m",  # Blue
                }
                reset = "\033[0m"
                color = colors.get(status, "")
                print(f"{color}{status}: {count}{reset}")
        
        # Print failed tests
        failures = [r for r in self.results if r["status"] == "FAIL"]
        if failures:
            print(f"\n{chr(10007)} FAILED TESTS:")
            for failure in failures:
                print(f"  • {failure['system']} - {failure['test']}: {failure['message']}")
        
        # Print warnings
        warnings = [r for r in self.results if r["status"] == "WARN"]
        if warnings:
            print(f"\n⚠ WARNINGS:")
            for warning in warnings:
                print(f"  • {warning['system']} - {warning['test']}: {warning['message']}")
        
        # Final status
        if failures:
            print(f"\n❌ VALIDATION FAILED: {len(failures)} critical issues found")
            return False
        elif warnings:
            print(f"\n⚠️  VALIDATION PASSED WITH WARNINGS: {len(warnings)} issues to review")
            return True
        else:
            print(f"\n✅ VALIDATION PASSED: All tests successful")
            return True


async def main():
    """Run enterprise setup validation."""
    print("Project Maestro - Enterprise Setup Validation")
    print("=" * 50)
    
    validator = EnterpriseValidator()
    
    # Run all validations
    validator.validate_basic_config()
    validator.validate_enterprise_config()
    
    await validator.validate_jira_connectivity()
    await validator.validate_slack_connectivity() 
    await validator.validate_confluence_connectivity()
    
    validator.validate_vector_store()
    
    # Print summary and exit with appropriate code
    success = validator.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())