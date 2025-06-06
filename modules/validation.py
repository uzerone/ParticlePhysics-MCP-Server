import re
import logging
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from functools import wraps
from asyncio import Lock
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_value: Any = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class SecurityLogger:
    """Security event logger."""
    
    def __init__(self):
        self.security_logger = logging.getLogger("pdg_security")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - SECURITY - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.security_logger.addHandler(handler)
        self.security_logger.setLevel(logging.WARNING)
    
    def log_validation_failure(self, input_type: str, value: str, errors: List[str], source_ip: str = "unknown"):
        """Log validation failure."""
        if config.security.log_security_events:
            self.security_logger.warning(
                f"Validation failure - Type: {input_type}, Value: {value[:50]}..., "
                f"Errors: {errors}, Source: {source_ip}"
            )
    
    def log_rate_limit_exceeded(self, source_ip: str, endpoint: str):
        """Log rate limit exceeded."""
        if config.security.log_security_events:
            self.security_logger.warning(
                f"Rate limit exceeded - Source: {source_ip}, Endpoint: {endpoint}"
            )
    
    def log_suspicious_activity(self, activity_type: str, details: str, source_ip: str = "unknown"):
        """Log suspicious activity."""
        if config.security.log_security_events:
            self.security_logger.error(
                f"Suspicious activity - Type: {activity_type}, Details: {details}, Source: {source_ip}"
            )


class RateLimiter:
    """Token bucket rate limiter with per-IP tracking."""
    
    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = Lock()
    
    async def is_allowed(self, identifier: str = "global") -> bool:
        """Check if request is allowed under rate limit."""
        if not config.rate_limit.enabled:
            return True
        
        async with self.lock:
            now = time.time()
            window_start = now - 1.0  # 1 second window
            
            # Clean old requests
            while self.requests[identifier] and self.requests[identifier][0] < window_start:
                self.requests[identifier].popleft()
            
            # Check rate limit
            if len(self.requests[identifier]) >= config.rate_limit.requests_per_second:
                return False
            
            # Add current request
            self.requests[identifier].append(now)
            return True
    
    async def get_remaining_capacity(self, identifier: str = "global") -> int:
        """Get remaining capacity for identifier."""
        if not config.rate_limit.enabled:
            return float('inf')
        
        async with self.lock:
            now = time.time()
            window_start = now - 1.0
            
            # Clean old requests
            while self.requests[identifier] and self.requests[identifier][0] < window_start:
                self.requests[identifier].popleft()
            
            return max(0, int(config.rate_limit.requests_per_second) - len(self.requests[identifier]))


class InputValidator:
    """Comprehensive input validator with security features."""
    
    def __init__(self):
        self.security_logger = SecurityLogger()
        self.rate_limiter = RateLimiter()
        
        # Compile regex patterns for performance
        self.allowed_chars_pattern = re.compile(config.security.allowed_characters_pattern)
        self.particle_name_pattern = re.compile(r"^[a-zA-Z0-9\-\+\*\(\)\[\]\'\"_/]+$")
        self.pdg_id_pattern = re.compile(r"^[SMGTDE]\d{3}(/\d{4})?$")
        self.mcid_pattern = re.compile(r"^\d+$")
        
        # Suspicious patterns
        self.suspicious_patterns = [
            re.compile(r"<script", re.IGNORECASE),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),
            re.compile(r"eval\s*\(", re.IGNORECASE),
            re.compile(r"exec\s*\(", re.IGNORECASE),
            re.compile(r"__\w+__"),  # Python dunder methods
        ]
    
    def validate_particle_name(self, name: str, source_ip: str = "unknown") -> ValidationResult:
        """Validate particle name input."""
        if not config.security.input_validation:
            return ValidationResult(is_valid=True, sanitized_value=name)
        
        errors = []
        warnings = []
        
        # Basic checks
        if not name or not isinstance(name, str):
            errors.append("Particle name must be a non-empty string")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Length check
        if len(name) > config.security.max_query_length:
            errors.append(f"Particle name too long (max {config.security.max_query_length} characters)")
        
        # Character validation
        if not self.particle_name_pattern.match(name):
            errors.append("Particle name contains invalid characters")
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern.search(name):
                errors.append("Particle name contains suspicious patterns")
                self.security_logger.log_suspicious_activity(
                    "suspicious_particle_name", f"Name: {name}", source_ip
                )
                break
        
        # Sanitize the name
        sanitized_name = name.strip()
        
        # Check for common typos and suggest corrections
        if len(sanitized_name) > 10:
            warnings.append("Unusually long particle name - verify spelling")
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized_name,
            errors=errors,
            warnings=warnings
        )
        
        if not result.is_valid:
            self.security_logger.log_validation_failure(
                "particle_name", name, errors, source_ip
            )
        
        return result
    
    def validate_pdg_id(self, pdg_id: str, source_ip: str = "unknown") -> ValidationResult:
        """Validate PDG identifier input."""
        if not config.security.input_validation:
            return ValidationResult(is_valid=True, sanitized_value=pdg_id)
        
        errors = []
        warnings = []
        
        if not pdg_id or not isinstance(pdg_id, str):
            errors.append("PDG ID must be a non-empty string")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Basic format validation
        if not self.pdg_id_pattern.match(pdg_id.upper()):
            errors.append("Invalid PDG ID format (expected: S008, M100, etc.)")
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern.search(pdg_id):
                errors.append("PDG ID contains suspicious patterns")
                self.security_logger.log_suspicious_activity(
                    "suspicious_pdg_id", f"ID: {pdg_id}", source_ip
                )
                break
        
        sanitized_id = pdg_id.strip().upper()
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized_id,
            errors=errors,
            warnings=warnings
        )
        
        if not result.is_valid:
            self.security_logger.log_validation_failure(
                "pdg_id", pdg_id, errors, source_ip
            )
        
        return result
    
    def validate_mcid(self, mcid: Union[str, int], source_ip: str = "unknown") -> ValidationResult:
        """Validate Monte Carlo ID input."""
        if not config.security.input_validation:
            return ValidationResult(is_valid=True, sanitized_value=mcid)
        
        errors = []
        warnings = []
        
        # Convert to string for validation
        mcid_str = str(mcid)
        
        if not mcid_str or mcid_str == "None":
            errors.append("Monte Carlo ID must be provided")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check if it's a valid integer
        try:
            mcid_int = int(mcid_str)
            if mcid_int < 0:
                errors.append("Monte Carlo ID must be non-negative")
            if mcid_int > 999999:  # Reasonable upper limit
                warnings.append("Very large Monte Carlo ID - verify correctness")
        except ValueError:
            errors.append("Monte Carlo ID must be a valid integer")
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern.search(mcid_str):
                errors.append("Monte Carlo ID contains suspicious patterns")
                self.security_logger.log_suspicious_activity(
                    "suspicious_mcid", f"MCID: {mcid_str}", source_ip
                )
                break
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=mcid_int if len(errors) == 0 else mcid_str,
            errors=errors,
            warnings=warnings
        )
        
        if not result.is_valid:
            self.security_logger.log_validation_failure(
                "mcid", mcid_str, errors, source_ip
            )
        
        return result
    
    def validate_query_string(self, query: str, source_ip: str = "unknown") -> ValidationResult:
        """Validate general query string input."""
        if not config.security.input_validation:
            return ValidationResult(is_valid=True, sanitized_value=query)
        
        errors = []
        warnings = []
        
        if not query or not isinstance(query, str):
            errors.append("Query must be a non-empty string")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Length validation
        if len(query) > config.security.max_query_length:
            errors.append(f"Query too long (max {config.security.max_query_length} characters)")
        
        # Character validation
        if not self.allowed_chars_pattern.match(query):
            errors.append("Query contains invalid characters")
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern.search(query):
                errors.append("Query contains suspicious patterns")
                self.security_logger.log_suspicious_activity(
                    "suspicious_query", f"Query: {query}", source_ip
                )
                break
        
        sanitized_query = query.strip()
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized_query,
            errors=errors,
            warnings=warnings
        )
        
        if not result.is_valid:
            self.security_logger.log_validation_failure(
                "query", query, errors, source_ip
            )
        
        return result
    
    def validate_numeric_parameter(
        self, 
        value: Union[str, int, float], 
        param_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        source_ip: str = "unknown"
    ) -> ValidationResult:
        """Validate numeric parameter."""
        if not config.security.input_validation:
            return ValidationResult(is_valid=True, sanitized_value=value)
        
        errors = []
        warnings = []
        
        try:
            if isinstance(value, str):
                numeric_value = float(value)
            else:
                numeric_value = float(value)
            
            if min_value is not None and numeric_value < min_value:
                errors.append(f"{param_name} must be >= {min_value}")
            
            if max_value is not None and numeric_value > max_value:
                errors.append(f"{param_name} must be <= {max_value}")
            
        except (ValueError, TypeError):
            errors.append(f"{param_name} must be a valid number")
            numeric_value = value
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=numeric_value if len(errors) == 0 else value,
            errors=errors,
            warnings=warnings
        )
        
        if not result.is_valid:
            self.security_logger.log_validation_failure(
                f"numeric_{param_name}", str(value), errors, source_ip
            )
        
        return result
    
    async def check_rate_limit(self, identifier: str = "global", endpoint: str = "unknown") -> bool:
        """Check rate limit for request."""
        is_allowed = await self.rate_limiter.is_allowed(identifier)
        
        if not is_allowed:
            self.security_logger.log_rate_limit_exceeded(identifier, endpoint)
        
        return is_allowed


# Global validator instance
validator = InputValidator()


def validate_input(input_type: str, **kwargs):
    """Decorator for input validation."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **func_kwargs):
            # Extract source IP if available (would need to be passed in arguments)
            source_ip = func_kwargs.get('source_ip', 'unknown')
            
            # Check rate limit first
            if not await validator.check_rate_limit(source_ip, func.__name__):
                raise ValueError("Rate limit exceeded. Please slow down your requests.")
            
            # Validate inputs based on type
            if input_type == "particle_name" and "particle_name" in func_kwargs:
                result = validator.validate_particle_name(
                    func_kwargs["particle_name"], source_ip
                )
                if not result.is_valid:
                    raise ValueError(f"Invalid particle name: {', '.join(result.errors)}")
                func_kwargs["particle_name"] = result.sanitized_value
            
            elif input_type == "pdg_id" and "pdgid" in func_kwargs:
                result = validator.validate_pdg_id(func_kwargs["pdgid"], source_ip)
                if not result.is_valid:
                    raise ValueError(f"Invalid PDG ID: {', '.join(result.errors)}")
                func_kwargs["pdgid"] = result.sanitized_value
            
            elif input_type == "query" and "query" in func_kwargs:
                result = validator.validate_query_string(func_kwargs["query"], source_ip)
                if not result.is_valid:
                    raise ValueError(f"Invalid query: {', '.join(result.errors)}")
                func_kwargs["query"] = result.sanitized_value
            
            return await func(*args, **func_kwargs)
        return wrapper
    return decorator


def sanitize_output(data: Any, max_size: int = None) -> Any:
    """Sanitize output data to prevent information disclosure."""
    if max_size is None:
        max_size = config.performance.max_response_size
    
    # Convert to JSON string to check size
    import json
    try:
        json_str = json.dumps(data)
        if len(json_str.encode('utf-8')) > max_size:
            return {
                "error": "Response too large",
                "message": f"Response exceeds maximum size of {max_size} bytes",
                "truncated": True
            }
    except (TypeError, ValueError):
        # If we can't serialize it, return a safe representation
        return {
            "error": "Serialization error",
            "message": "Unable to serialize response data",
            "type": type(data).__name__
        }
    
    return data 