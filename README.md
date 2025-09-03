# Project Maestro

**AI Agent-based Game Prototyping Automation System**

Project Maestro is a sophisticated multi-agent orchestration system that automatically converts game design documents into playable game prototypes using AI agents and Unity integration.

[![Tests](https://github.com/your-org/project-maestro/workflows/tests/badge.svg)](https://github.com/your-org/project-maestro/actions)
[![Coverage](https://codecov.io/gh/your-org/project-maestro/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/project-maestro)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🚀 Features

- **Intelligent Game Design Document Parsing**: Advanced NLP-powered analysis of game design requirements
- **Multi-Agent Orchestration**: Specialized AI agents for different aspects of game development
- **Automated Asset Generation**: AI-powered creation of code, art, audio, and level assets
- **Unity Integration**: Seamless Unity project creation and cross-platform building
- **Real-time Monitoring**: Comprehensive system monitoring and performance analytics
- **Scalable Architecture**: Event-driven microservices with Redis message queuing
- **RESTful API**: Complete API for integration and external tool support

## 🏗️ Architecture

### Agent Types

- **🎭 Orchestrator**: Master agent coordinating the entire workflow
- **💻 Codex**: C# game code generation and Unity scripting
- **🎨 Canvas**: Visual asset generation using Stable Diffusion
- **🎵 Sonata**: Audio and music generation
- **🗺️ Labyrinth**: Level design and gameplay progression
- **🔨 Builder**: Unity integration and cross-platform building

### Core Systems

- **Event-Driven Messaging**: Redis-powered async communication
- **Multi-Backend Storage**: Support for MinIO, S3, and local storage
- **Comprehensive Monitoring**: Real-time metrics, alerting, and health checks
- **Advanced Error Handling**: Circuit breakers, retry mechanisms, and recovery strategies

## 📋 Requirements

- **Python**: 3.9 or higher
- **Unity**: 2023.2.0f1 or later
- **Redis**: 6.0 or higher
- **Storage**: MinIO, S3, or local filesystem

### API Keys (Optional)

- OpenAI API key for GPT models
- Anthropic API key for Claude models
- Stable Diffusion API access for image generation

## 🛠️ Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/project-maestro.git
cd project-maestro

# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize the project
maestro init

# Start the API server
maestro server start
```

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev,test]"

# Run tests
python run_tests.py

# Start with development settings
export MAESTRO_ENVIRONMENT=development
maestro server start --reload
```

## 🚦 Quick Start Guide

### 1. Create a Game Design Document

Create a markdown file describing your game:

```markdown
# My Platformer Game

## Game Overview
- **Genre**: Platformer
- **Platform**: Mobile (Android/iOS)
- **Art Style**: Pixel art

## Core Gameplay
- Player controls a character that can move left/right and jump
- Collect coins and avoid enemies
- Reach the end of each level

## Characters
- **Hero**: Main player character with jump and move abilities
- **Goomba**: Simple enemy that moves left and right

## Levels
- **Level 1**: Green hills with 3 platforms and 2 enemies
```

### 2. Generate Your Game

Using the CLI:

```bash
maestro project create "My Platformer" \
  --document-file game_design.md \
  --description "A simple platformer game"
```

Using the API:

```bash
curl -X POST "http://localhost:8000/api/v1/projects/" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My Platformer",
    "description": "A simple platformer game",
    "game_design_document": "# My Game..."
  }'
```

### 3. Monitor Progress

```bash
# Check project status
maestro project status <project-id>

# Monitor in real-time
maestro project status <project-id> --watch

# View system metrics
curl http://localhost:8000/metrics
```

## 🔧 Configuration

### Environment Variables

```bash
# Core Configuration
MAESTRO_ENVIRONMENT=production
MAESTRO_DEBUG=false
MAESTRO_LOG_LEVEL=INFO

# API Configuration
MAESTRO_API_HOST=0.0.0.0
MAESTRO_API_PORT=8000
MAESTRO_API_WORKERS=4

# Database
MAESTRO_DATABASE_URL=postgresql://user:pass@localhost/maestro

# Redis
MAESTRO_REDIS_URL=redis://localhost:6379/0

# Storage
MAESTRO_STORAGE_TYPE=minio  # minio, s3, local
MAESTRO_MINIO_ENDPOINT=localhost:9000
MAESTRO_MINIO_ACCESS_KEY=minioaccess
MAESTRO_MINIO_SECRET_KEY=miniosecret

# AI Services
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
STABLE_DIFFUSION_API_KEY=your_sd_key

# Unity
MAESTRO_UNITY_PATH=/Applications/Unity/Hub/Editor/2023.2.0f1
```

### Advanced Configuration

See [docs/configuration.md](docs/configuration.md) for detailed configuration options.

## 📖 API Documentation

### Core Endpoints

- `GET /health` - System health check
- `GET /metrics` - System metrics and monitoring
- `POST /api/v1/projects/` - Create new game project
- `GET /api/v1/projects/{id}` - Get project details
- `POST /api/v1/builds/` - Create game build
- `GET /api/v1/agents/` - List agent status

### Interactive Documentation

Start the server and visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Testing

### Run All Tests

```bash
python run_tests.py
```

### Run Specific Test Types

```bash
# Unit tests only
python run_tests.py --unit

# Integration tests
python run_tests.py --integration

# Performance tests  
python run_tests.py --performance

# API tests
python run_tests.py --api

# Code quality checks
python run_tests.py --lint
```

### Test Coverage

Tests maintain >80% coverage. View coverage report:

```bash
python run_tests.py --report
open test_reports/coverage/index.html
```

## 📊 Monitoring

### System Health

```bash
# Check overall system health
curl http://localhost:8000/health

# Get detailed metrics
curl http://localhost:8000/metrics

# Agent-specific metrics
curl http://localhost:8000/metrics/agents/orchestrator
```

### Monitoring Dashboard

Project Maestro includes built-in monitoring:

- **System Metrics**: CPU, memory, disk usage
- **Agent Performance**: Task completion rates, response times
- **Error Tracking**: Comprehensive error categorization and recovery
- **Real-time Alerts**: Configurable alerting for system issues

## 🔍 Troubleshooting

### Common Issues

**Agent Not Responding**
```bash
# Check agent status
maestro agent status orchestrator

# Restart agents
maestro server restart
```

**Build Failures**
```bash
# Check Unity path
maestro config

# View build logs
curl http://localhost:8000/api/v1/builds/{build-id}/logs
```

**High Memory Usage**
```bash
# Check system metrics
curl http://localhost:8000/metrics

# View error statistics
curl http://localhost:8000/api/v1/analytics/errors/summary
```

### Debug Mode

```bash
export MAESTRO_DEBUG=true
export MAESTRO_LOG_LEVEL=DEBUG
maestro server start
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Code Quality Standards

- **Tests**: All new code must include tests
- **Coverage**: Maintain >80% test coverage
- **Documentation**: Update documentation for new features
- **Code Style**: Follow PEP 8 and use Black formatting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the multi-agent framework
- **Unity** for game engine integration  
- **OpenAI & Anthropic** for AI model access
- **FastAPI** for the REST API framework
- **Redis** for message queuing

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/project-maestro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/project-maestro/discussions)

## 🗺️ Roadmap

### Version 1.0 (Current)
- ✅ Multi-agent orchestration
- ✅ Basic asset generation
- ✅ Unity integration
- ✅ REST API

### Version 1.1 (Planned)
- 🔄 Advanced AI models integration
- 🔄 Web-based UI dashboard
- 🔄 Plugin system for custom agents
- 🔄 Enhanced analytics and reporting

### Version 2.0 (Future)
- 📋 Visual scripting support
- 📋 Multiplayer game templates
- 📋 Advanced AI behavior trees
- 📋 Cloud deployment automation

---

**Made with ❤️ by the Project Maestro team**