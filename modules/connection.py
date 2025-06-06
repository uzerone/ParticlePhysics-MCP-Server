import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Connection pool statistics."""

    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    retry_attempts: int = 0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None


class PDGConnection:
    """Wrapper for PDG API connection with health checking."""

    def __init__(self, api_instance):
        self.api = api_instance
        self.created_at = time.time()
        self.last_used = time.time()
        self.is_healthy = True
        self.use_count = 0

    def touch(self):
        """Update last used timestamp."""
        self.last_used = time.time()
        self.use_count += 1

    async def health_check(self) -> bool:
        """Check if connection is still healthy."""
        try:
            # Simple health check - try to access a basic function
            # This would depend on your PDG API implementation
            if hasattr(self.api, "get_database_info"):
                await asyncio.get_event_loop().run_in_executor(
                    None, self.api.get_database_info
                )
            self.is_healthy = True
            logger.debug("PDG connection health check passed")
            return True
        except Exception as e:
            logger.warning(f"PDG connection health check failed: {e}")
            self.is_healthy = False
            return False

    @property
    def age_seconds(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_used


class PDGConnectionPool:
    """Thread-safe connection pool for PDG API connections."""

    def __init__(self, max_connections: int = None, max_idle_time: float = 300):
        self.max_connections = max_connections or config.connection.pool_size
        self.max_idle_time = max_idle_time

        self._connections: List[PDGConnection] = []
        self._lock = asyncio.Lock()
        self._stats = ConnectionStats()

        # Thread pool for blocking PDG operations
        self._executor = ThreadPoolExecutor(max_workers=self.max_connections)

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task for idle connections."""

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # Cleanup every minute
                    await self._cleanup_idle_connections()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Connection cleanup error: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def _cleanup_idle_connections(self):
        """Remove idle connections that exceed max_idle_time."""
        async with self._lock:
            current_time = time.time()
            connections_to_remove = []

            for connection in self._connections:
                if current_time - connection.last_used > self.max_idle_time:
                    connections_to_remove.append(connection)

            for connection in connections_to_remove:
                self._connections.remove(connection)
                self._stats.active_connections -= 1
                logger.debug(
                    f"Removed idle PDG connection (idle for {connection.idle_seconds:.1f}s)"
                )

    async def _create_connection(self) -> PDGConnection:
        """Create a new PDG API connection with retry logic."""
        retry_count = 0
        max_retries = config.connection.retry_attempts
        base_delay = config.connection.retry_delay

        while retry_count <= max_retries:
            try:
                # Import and connect to PDG API
                import pdg

                # Run the blocking connection operation in executor
                api_instance = await asyncio.get_event_loop().run_in_executor(
                    self._executor, pdg.connect
                )

                connection = PDGConnection(api_instance)

                self._stats.total_connections += 1
                self._stats.last_success = time.time()

                logger.info(f"Created new PDG connection (attempt {retry_count + 1})")
                return connection

            except ImportError as e:
                logger.error(
                    "PDG package not installed. Please install it using: pip install pdg"
                )
                raise Exception("PDG package not installed") from e

            except Exception as e:
                retry_count += 1
                self._stats.retry_attempts += 1
                self._stats.last_failure = time.time()

                if retry_count > max_retries:
                    self._stats.failed_connections += 1
                    logger.error(
                        f"Failed to create PDG connection after {max_retries} retries: {e}"
                    )
                    raise Exception(f"Failed to connect to PDG database: {e}") from e

                # Exponential backoff
                delay = base_delay * (
                    config.connection.retry_backoff ** (retry_count - 1)
                )
                logger.warning(
                    f"PDG connection failed (attempt {retry_count}), retrying in {delay}s: {e}"
                )
                await asyncio.sleep(delay)

        raise Exception("Unexpected exit from retry loop")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool (context manager)."""
        connection = None

        try:
            async with self._lock:
                # Try to find an existing healthy connection
                for conn in self._connections:
                    if conn.is_healthy:
                        conn.touch()
                        connection = conn
                        break

                # If no healthy connection found, create a new one
                if connection is None:
                    if len(self._connections) < self.max_connections:
                        connection = await self._create_connection()
                        self._connections.append(connection)
                        self._stats.active_connections += 1
                    else:
                        # Pool is full, wait and retry or use oldest connection
                        if self._connections:
                            connection = min(
                                self._connections, key=lambda c: c.last_used
                            )
                            connection.touch()
                        else:
                            raise Exception(
                                "No PDG connections available and pool is full"
                            )

            # Yield the connection for use
            yield connection

        except Exception as e:
            logger.error(f"Error getting PDG connection: {e}")
            if connection:
                connection.is_healthy = False
            raise

        finally:
            # Connection is automatically returned to pool
            pass

    async def get_connection_sync(self) -> PDGConnection:
        """Get a connection synchronously (for backward compatibility)."""
        async with self._lock:
            # Try to find an existing healthy connection
            for connection in self._connections:
                if connection.is_healthy:
                    connection.touch()
                    return connection

            # Create new connection if pool not full
            if len(self._connections) < self.max_connections:
                connection = await self._create_connection()
                self._connections.append(connection)
                self._stats.active_connections += 1
                return connection

            # Pool is full, return least recently used connection
            if self._connections:
                connection = min(self._connections, key=lambda c: c.last_used)
                connection.touch()
                return connection

            raise Exception("No PDG connections available")

    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all connections."""
        async with self._lock:
            healthy_count = 0
            unhealthy_count = 0

            for connection in self._connections:
                if await connection.health_check():
                    healthy_count += 1
                else:
                    unhealthy_count += 1

            return {
                "total_connections": len(self._connections),
                "healthy_connections": healthy_count,
                "unhealthy_connections": unhealthy_count,
                "pool_utilization": len(self._connections) / self.max_connections,
            }

    async def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        async with self._lock:
            pool_utilization = (
                len(self._connections) / self.max_connections
                if self.max_connections > 0
                else 0
            )

            return {
                "pool_size": self.max_connections,
                "active_connections": len(self._connections),
                "pool_utilization": pool_utilization,
                "total_connections_created": self._stats.total_connections,
                "failed_connections": self._stats.failed_connections,
                "retry_attempts": self._stats.retry_attempts,
                "last_success": self._stats.last_success,
                "last_failure": self._stats.last_failure,
            }

    async def close_all(self):
        """Close all connections in the pool."""
        async with self._lock:
            self._connections.clear()
            self._stats.active_connections = 0

        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

        # Shutdown executor
        self._executor.shutdown(wait=True)

        logger.info("Closed all PDG connections")

    def __del__(self):
        """Cleanup when pool is destroyed."""
        if (
            hasattr(self, "_cleanup_task")
            and self._cleanup_task
            and not self._cleanup_task.done()
        ):
            self._cleanup_task.cancel()
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


# Global connection pool instance
connection_pool = PDGConnectionPool()


async def get_pdg_connection():
    """Get a PDG connection from the pool (backward compatibility function)."""
    return await connection_pool.get_connection_sync()


async def get_connection_pool_stats() -> Dict[str, Any]:
    """Get connection pool statistics."""
    return await connection_pool.get_stats()


async def health_check_connections() -> Dict[str, Any]:
    """Perform health check on all connections."""
    return await connection_pool.health_check_all()


# Context manager for getting connections
@asynccontextmanager
async def pdg_connection():
    """Context manager for getting a PDG connection."""
    async with connection_pool.get_connection() as connection:
        yield connection.api
