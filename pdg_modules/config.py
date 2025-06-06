"""
PDG MCP Server Configuration Module

Centralized configuration management with environment variable support,
caching settings, connection pooling, and runtime configuration.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    enabled: bool = True
    ttl_seconds: int = 300  # 5 minutes default
    max_size: int = 1000
    backend: str = "memory"  # memory, redis
    redis_url: Optional[str] = None


@dataclass
class ConnectionConfig:
    """PDG API connection configuration."""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    enabled: bool = True
    requests_per_second: float = 10.0
    burst_limit: int = 50


@dataclass
class SecurityConfig:
    """Security configuration."""
    input_validation: bool = True
    max_query_length: int = 1000
    allowed_characters_pattern: str = r"^[a-zA-Z0-9\-\+\*\(\)\[\]\{\}\.\,\s_/]*$"
    log_security_events: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class PerformanceConfig:
    """Performance optimization settings."""
    max_response_size: int = 50 * 1024 * 1024  # 50MB
    pagination_default_limit: int = 50
    pagination_max_limit: int = 1000
    async_timeout: float = 30.0


@dataclass
class PDGServerConfig:
    """Main configuration class for PDG MCP Server."""
    
    # Component configurations (required, no defaults)
    cache: CacheConfig
    connection: ConnectionConfig
    rate_limit: RateLimitConfig
    security: SecurityConfig
    logging: LoggingConfig
    performance: PerformanceConfig
    features: Dict[str, bool]
    
    # Environment (optional, with defaults)
    environment: str = "development"
    debug: bool = False
    
    # Server settings (optional, with defaults)
    server_name: str = "pdg-mcp-server"
    server_version: str = "1.0.0"
    
    @classmethod
    def from_environment(cls) -> 'PDGServerConfig':
        """Create configuration from environment variables."""
        
        # Cache configuration
        cache = CacheConfig(
            enabled=_get_bool_env("PDG_CACHE_ENABLED", True),
            ttl_seconds=_get_int_env("PDG_CACHE_TTL_SECONDS", 300),
            max_size=_get_int_env("PDG_CACHE_MAX_SIZE", 1000),
            backend=os.getenv("PDG_CACHE_BACKEND", "memory"),
            redis_url=os.getenv("PDG_CACHE_REDIS_URL"),
        )
        
        # Connection configuration
        connection = ConnectionConfig(
            pool_size=_get_int_env("PDG_POOL_SIZE", 10),
            max_overflow=_get_int_env("PDG_POOL_MAX_OVERFLOW", 20),
            pool_timeout=_get_int_env("PDG_POOL_TIMEOUT", 30),
            retry_attempts=_get_int_env("PDG_RETRY_ATTEMPTS", 3),
            retry_delay=_get_float_env("PDG_RETRY_DELAY", 1.0),
            retry_backoff=_get_float_env("PDG_RETRY_BACKOFF", 2.0),
        )
        
        # Rate limiting configuration
        rate_limit = RateLimitConfig(
            enabled=_get_bool_env("PDG_RATE_LIMIT_ENABLED", True),
            requests_per_second=_get_float_env("PDG_RATE_LIMIT_RPS", 10.0),
            burst_limit=_get_int_env("PDG_RATE_LIMIT_BURST", 50),
        )
        
        # Security configuration
        security = SecurityConfig(
            input_validation=_get_bool_env("PDG_INPUT_VALIDATION", True),
            max_query_length=_get_int_env("PDG_MAX_QUERY_LENGTH", 1000),
            allowed_characters_pattern=os.getenv(
                "PDG_ALLOWED_CHARS_PATTERN",
                r"^[a-zA-Z0-9\-\+\*\(\)\[\]\{\}\.\,\s_/]*$"
            ),
            log_security_events=_get_bool_env("PDG_LOG_SECURITY_EVENTS", True),
        )
        
        # Logging configuration
        logging_config = LoggingConfig(
            level=os.getenv("PDG_LOG_LEVEL", "INFO"),
            format=os.getenv(
                "PDG_LOG_FORMAT",
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
            file_path=os.getenv("PDG_LOG_FILE_PATH"),
            max_file_size=_get_int_env("PDG_LOG_MAX_FILE_SIZE", 10 * 1024 * 1024),
            backup_count=_get_int_env("PDG_LOG_BACKUP_COUNT", 5),
        )
        
        # Performance configuration
        performance = PerformanceConfig(
            max_response_size=_get_int_env("PDG_MAX_RESPONSE_SIZE", 50 * 1024 * 1024),
            pagination_default_limit=_get_int_env("PDG_PAGINATION_DEFAULT_LIMIT", 50),
            pagination_max_limit=_get_int_env("PDG_PAGINATION_MAX_LIMIT", 1000),
            async_timeout=_get_float_env("PDG_ASYNC_TIMEOUT", 30.0),
        )
        
        # Feature flags
        features = {
            "caching": _get_bool_env("PDG_FEATURE_CACHING", True),
            "rate_limiting": _get_bool_env("PDG_FEATURE_RATE_LIMITING", True),
            "input_validation": _get_bool_env("PDG_FEATURE_INPUT_VALIDATION", True),
            "metrics": _get_bool_env("PDG_FEATURE_METRICS", False),
            "health_checks": _get_bool_env("PDG_FEATURE_HEALTH_CHECKS", True),
        }
        
        return cls(
            environment=os.getenv("PDG_ENVIRONMENT", "development"),
            debug=_get_bool_env("PDG_DEBUG", False),
            server_name=os.getenv("PDG_SERVER_NAME", "pdg-mcp-server"),
            server_version=os.getenv("PDG_SERVER_VERSION", "1.0.0"),
            cache=cache,
            connection=connection,
            rate_limit=rate_limit,
            security=security,
            logging=logging_config,
            performance=performance,
            features=features,
        )
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate cache configuration
            if self.cache.ttl_seconds <= 0:
                raise ValueError("Cache TTL must be positive")
            
            if self.cache.max_size <= 0:
                raise ValueError("Cache max size must be positive")
            
            # Validate connection configuration
            if self.connection.pool_size <= 0:
                raise ValueError("Connection pool size must be positive")
            
            if self.connection.retry_attempts < 0:
                raise ValueError("Retry attempts cannot be negative")
            
            # Validate rate limiting
            if self.rate_limit.requests_per_second <= 0:
                raise ValueError("Rate limit RPS must be positive")
            
            # Validate security settings
            if self.security.max_query_length <= 0:
                raise ValueError("Max query length must be positive")
            
            # Validate performance settings
            if self.performance.pagination_default_limit <= 0:
                raise ValueError("Default pagination limit must be positive")
            
            if self.performance.pagination_max_limit < self.performance.pagination_default_limit:
                raise ValueError("Max pagination limit must be >= default limit")
            
            logger.info("Configuration validation successful")
            return True
            
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging/debugging."""
        config_dict = {
            "environment": self.environment,
            "debug": self.debug,
            "server_name": self.server_name,
            "server_version": self.server_version,
            "cache": {
                "enabled": self.cache.enabled,
                "ttl_seconds": self.cache.ttl_seconds,
                "max_size": self.cache.max_size,
                "backend": self.cache.backend,
                # Don't log sensitive redis_url
            },
            "connection": {
                "pool_size": self.connection.pool_size,
                "max_overflow": self.connection.max_overflow,
                "pool_timeout": self.connection.pool_timeout,
                "retry_attempts": self.connection.retry_attempts,
            },
            "rate_limit": {
                "enabled": self.rate_limit.enabled,
                "requests_per_second": self.rate_limit.requests_per_second,
                "burst_limit": self.rate_limit.burst_limit,
            },
            "security": {
                "input_validation": self.security.input_validation,
                "max_query_length": self.security.max_query_length,
                "log_security_events": self.security.log_security_events,
            },
            "features": self.features,
        }
        return config_dict


def _get_bool_env(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    else:
        return default


def _get_int_env(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"Invalid integer value for {key}, using default: {default}")
        return default


def _get_float_env(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"Invalid float value for {key}, using default: {default}")
        return default


# Global configuration instance
config = PDGServerConfig.from_environment()

# Validate configuration on import
if not config.validate():
    logger.error("Configuration validation failed. Some features may not work correctly.") 