# Adeloop Backend - Refactored Architecture

This document describes the refactored backend architecture for Adeloop, which has been reorganized for better maintainability, scalability, and code organization.

## 🏗️ Architecture Overview

The backend has been refactored from a monolithic structure to a modular, well-organized architecture:

```
backend/
├── main.py                 # Clean entry point
├── config/                 # Configuration management
│   ├── __init__.py
│   └── settings.py         # Environment variables and app settings
├── api/                    # API layer
│   ├── __init__.py
│   ├── middleware.py       # CORS, logging, security middleware
│   └── routes/             # API route modules
│       ├── __init__.py
│       ├── execution.py    # Python/JS execution endpoints
│       ├── monitoring.py   # Health and monitoring endpoints
│       └── streamlit.py    # Streamlit app endpoints
├── services/               # Business logic layer
│   ├── __init__.py
│   ├── kernel/             # Jupyter kernel services
│   │   ├── __init__.py
│   │   ├── jupyter_kernel.py
│   │   └── kernel_manager.py
│   ├── streamlit/          # Streamlit app management
│   │   ├── __init__.py
│   │   └── streamlit_manager.py
│   └── monitoring/         # Server monitoring services
│       ├── __init__.py
│       └── server_monitoring.py
├── models/                 # Data models
│   ├── __init__.py
│   └── requests.py         # Pydantic request models
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── responses.py        # Response formatting utilities
│   └── helpers.py          # General helper functions
└── tests/                  # Test modules
    ├── __init__.py
    └── test_server.py      # Server testing suite
```

## 🔧 Key Improvements

### 1. **Separation of Concerns**
- **API Layer**: Clean route definitions with proper error handling
- **Service Layer**: Business logic separated from API concerns
- **Models**: Centralized data validation and serialization
- **Utils**: Reusable utility functions

### 2. **Configuration Management**
- Centralized configuration in `config/settings.py`
- Environment variable handling
- Cloud platform detection
- Application metadata management

### 3. **Middleware Architecture**
- CORS configuration
- Request/response logging
- Security headers
- Rate limiting
- Global error handling

### 4. **Modular Services**
- **Kernel Services**: Jupyter kernel management and execution
- **Streamlit Services**: App creation and lifecycle management
- **Monitoring Services**: System health and metrics

### 5. **Enhanced Testing**
- Comprehensive test suite in `tests/test_server.py`
- Automated endpoint testing
- Performance monitoring
- Health check validation

## 📋 API Endpoints

### Execution Endpoints (`/api`)
- `POST /api/execute-jupyter` - Python code execution via Jupyter kernel
- `POST /api/execute` - Legacy endpoint (redirects to Jupyter)
- `POST /api/python/continue` - Continue execution with user input
- `POST /api/python/reset` - Reset Python kernel
- `GET /api/python/status` - Get kernel status
- `POST /api/python/install` - Install Python packages
- `GET /api/python/variables` - Get kernel variables
- `POST /api/python/clear-variables` - Clear kernel variables

### Monitoring Endpoints
- `GET /health` - Basic health check
- `POST /health/detailed` - Detailed health check with metrics
- `GET /status` - Comprehensive server status
- `GET /system` - System information
- `GET /cpu` - CPU metrics
- `GET /memory` - Memory usage
- `GET /disk` - Disk usage
- `GET /network` - Network information
- `GET /process` - Process information
- `GET /packages` - Installed packages
- `GET /uptime` - Server uptime

### Streamlit Endpoints (`/api`)
- `POST /api/streamlit/create` - Create Streamlit app
- `DELETE /api/streamlit/{app_id}` - Stop Streamlit app
- `GET /api/streamlit/{app_id}/status` - Get app status
- `GET /api/streamlit/apps` - List all apps
- `GET /streamlit/{app_id}` - Serve app page (non-API)

## 🚀 Getting Started

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file with your configuration:
```env
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
NEXTJS_API_URL=http://localhost:3000
CLERK_SECRET_KEY=your_clerk_secret_key
```

### 3. Run the Server
```bash
python main.py
```

### 4. Run Tests
```bash
python tests/test_server.py
```

## 🔄 Migration from Old Structure

The refactoring maintains **100% backward compatibility**. All existing functionality has been preserved:

### Maintained Features
- ✅ Python code execution via Jupyter kernel
- ✅ Interactive input handling
- ✅ Streamlit app creation and management
- ✅ Server monitoring and health checks
- ✅ Package installation
- ✅ Variable management
- ✅ Plot and media handling
- ✅ Error handling and logging

### Legacy File Compatibility
Old files now redirect to new locations:
- `request_models.py` → `models/requests.py`
- `response_utils.py` → `utils/responses.py`
- `jupyter_kernel.py` → `services/kernel/jupyter_kernel.py`
- `kernel_manager.py` → `services/kernel/kernel_manager.py`
- `streamlit_manager.py` → `services/streamlit/streamlit_manager.py`
- `server_monitoring.py` → `services/monitoring/server_monitoring.py`
- `test_server.py` → `tests/test_server.py`

## 🛠️ Development Guidelines

### Adding New Endpoints
1. Create route functions in appropriate `api/routes/` module
2. Add business logic to relevant `services/` module
3. Use existing models or create new ones in `models/`
4. Add tests to `tests/test_server.py`

### Adding New Services
1. Create service module in `services/`
2. Follow existing patterns for error handling
3. Add configuration to `config/settings.py` if needed
4. Document the service functionality

### Code Style
- Follow existing patterns for consistency
- Use type hints for better code documentation
- Add comprehensive docstrings
- Handle errors gracefully with proper logging

## 📊 Performance Improvements

The refactored architecture provides:
- **Faster startup time** through lazy loading
- **Better memory management** with modular imports
- **Improved error handling** with centralized middleware
- **Enhanced monitoring** with detailed metrics
- **Cleaner logging** with structured output

## 🔒 Security Enhancements

- **Security headers** added via middleware
- **Rate limiting** implemented
- **Input validation** through Pydantic models
- **Error sanitization** to prevent information leakage
- **Environment variable protection**

## 📈 Monitoring and Observability

The refactored backend includes comprehensive monitoring:
- **Health checks** with detailed system metrics
- **Performance monitoring** with execution timing
- **Resource usage tracking** (CPU, memory, disk)
- **Package inventory** management
- **Uptime tracking** and reporting

## 🎯 Future Enhancements

The new architecture enables easy addition of:
- Database integration
- Caching layers
- Message queues
- Microservice decomposition
- API versioning
- Authentication/authorization improvements

---

**Note**: This refactoring maintains all existing functionality while providing a solid foundation for future development. All API endpoints remain unchanged, ensuring seamless integration with existing frontend code.
