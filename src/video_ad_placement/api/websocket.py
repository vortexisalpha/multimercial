"""
WebSocket Manager

This module provides WebSocket functionality for real-time communication
with clients, including job progress updates and connection management.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)


@dataclass
class WebSocketConnection:
    """WebSocket connection information."""
    websocket: WebSocket
    job_id: Optional[str]
    user_id: Optional[str]
    connected_at: float
    last_ping: float
    
    def is_active(self) -> bool:
        """Check if connection is active."""
        return self.websocket.client_state == WebSocketState.CONNECTED


class WebSocketManager:
    """Manages WebSocket connections and real-time communication."""
    
    def __init__(self):
        # Active connections
        self.connections: Dict[str, WebSocketConnection] = {}  # connection_id -> connection
        self.job_connections: Dict[str, Set[str]] = {}  # job_id -> set of connection_ids
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        
        # Connection management
        self.connection_counter = 0
        self.ping_interval = 30.0  # seconds
        self.ping_timeout = 10.0  # seconds
        self._ping_task: Optional[asyncio.Task] = None
        
        # Message queue for offline delivery
        self.message_queue: Dict[str, List[Dict[str, Any]]] = {}  # job_id -> messages
        
        logger.info("WebSocketManager initialized")
    
    def _generate_connection_id(self) -> str:
        """Generate unique connection ID."""
        self.connection_counter += 1
        return f"ws_{int(time.time())}_{self.connection_counter}"
    
    async def connect(self, websocket: WebSocket, job_id: Optional[str] = None, 
                     user_id: Optional[str] = None) -> str:
        """Accept WebSocket connection and register it."""
        try:
            await websocket.accept()
            
            connection_id = self._generate_connection_id()
            current_time = time.time()
            
            # Create connection record
            connection = WebSocketConnection(
                websocket=websocket,
                job_id=job_id,
                user_id=user_id,
                connected_at=current_time,
                last_ping=current_time
            )
            
            # Store connection
            self.connections[connection_id] = connection
            
            # Index by job ID
            if job_id:
                if job_id not in self.job_connections:
                    self.job_connections[job_id] = set()
                self.job_connections[job_id].add(connection_id)
                
                # Send queued messages for this job
                await self._send_queued_messages(job_id, connection_id)
            
            # Index by user ID
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
            
            # Start ping task if not running
            if not self._ping_task:
                self._ping_task = asyncio.create_task(self._ping_connections())
            
            # Send welcome message
            await self._send_to_connection(connection_id, {
                "type": "connection",
                "status": "connected",
                "connection_id": connection_id,
                "job_id": job_id,
                "timestamp": current_time
            })
            
            logger.info(f"WebSocket connected: {connection_id} (job: {job_id}, user: {user_id})")
            return connection_id
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            raise
    
    async def disconnect(self, connection_id: str):
        """Disconnect and cleanup WebSocket connection."""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        # Remove from indexes
        if connection.job_id and connection.job_id in self.job_connections:
            self.job_connections[connection.job_id].discard(connection_id)
            if not self.job_connections[connection.job_id]:
                del self.job_connections[connection.job_id]
        
        if connection.user_id and connection.user_id in self.user_connections:
            self.user_connections[connection.user_id].discard(connection_id)
            if not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]
        
        # Close WebSocket if still connected
        try:
            if connection.websocket.client_state == WebSocketState.CONNECTED:
                await connection.websocket.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket {connection_id}: {e}")
        
        # Remove connection
        del self.connections[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def disconnect_all(self):
        """Disconnect all WebSocket connections."""
        connection_ids = list(self.connections.keys())
        
        for connection_id in connection_ids:
            await self.disconnect(connection_id)
        
        # Stop ping task
        if self._ping_task:
            self._ping_task.cancel()
            self._ping_task = None
        
        logger.info("All WebSocket connections disconnected")
    
    async def send_update(self, job_id: str, data: Dict[str, Any]):
        """Send update to all connections subscribed to a job."""
        message = {
            "type": "job_update",
            "job_id": job_id,
            "timestamp": time.time(),
            **data
        }
        
        # Send to active connections
        if job_id in self.job_connections:
            connection_ids = list(self.job_connections[job_id])
            
            for connection_id in connection_ids:
                success = await self._send_to_connection(connection_id, message)
                if not success:
                    # Connection failed, remove it
                    await self.disconnect(connection_id)
        
        # Queue message for offline delivery
        if job_id not in self.message_queue:
            self.message_queue[job_id] = []
        
        self.message_queue[job_id].append(message)
        
        # Limit queue size (keep last 50 messages)
        if len(self.message_queue[job_id]) > 50:
            self.message_queue[job_id] = self.message_queue[job_id][-50:]
    
    async def send_to_user(self, user_id: str, data: Dict[str, Any]):
        """Send message to all connections for a user."""
        message = {
            "type": "user_message",
            "user_id": user_id,
            "timestamp": time.time(),
            **data
        }
        
        if user_id in self.user_connections:
            connection_ids = list(self.user_connections[user_id])
            
            for connection_id in connection_ids:
                success = await self._send_to_connection(connection_id, message)
                if not success:
                    await self.disconnect(connection_id)
    
    async def broadcast(self, data: Dict[str, Any], 
                       exclude_connections: Optional[Set[str]] = None):
        """Broadcast message to all active connections."""
        message = {
            "type": "broadcast",
            "timestamp": time.time(),
            **data
        }
        
        exclude_connections = exclude_connections or set()
        connection_ids = [
            cid for cid in self.connections.keys() 
            if cid not in exclude_connections
        ]
        
        for connection_id in connection_ids:
            success = await self._send_to_connection(connection_id, message)
            if not success:
                await self.disconnect(connection_id)
    
    async def _send_to_connection(self, connection_id: str, data: Dict[str, Any]) -> bool:
        """Send data to a specific connection."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        try:
            if connection.websocket.client_state == WebSocketState.CONNECTED:
                await connection.websocket.send_text(json.dumps(data))
                return True
            else:
                return False
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected during send: {connection_id}")
            return False
        except Exception as e:
            logger.error(f"Error sending to WebSocket {connection_id}: {e}")
            return False
    
    async def _send_queued_messages(self, job_id: str, connection_id: str):
        """Send queued messages for a job to a newly connected client."""
        if job_id not in self.message_queue:
            return
        
        messages = self.message_queue[job_id]
        
        for message in messages:
            success = await self._send_to_connection(connection_id, message)
            if not success:
                break
        
        logger.info(f"Sent {len(messages)} queued messages to {connection_id}")
    
    async def _ping_connections(self):
        """Periodically ping connections to keep them alive."""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                
                current_time = time.time()
                connection_ids = list(self.connections.keys())
                
                for connection_id in connection_ids:
                    connection = self.connections.get(connection_id)
                    if not connection:
                        continue
                    
                    # Check if connection is still active
                    if not connection.is_active():
                        await self.disconnect(connection_id)
                        continue
                    
                    # Send ping
                    ping_success = await self._send_to_connection(connection_id, {
                        "type": "ping",
                        "timestamp": current_time
                    })
                    
                    if ping_success:
                        connection.last_ping = current_time
                    else:
                        await self.disconnect(connection_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping task: {e}")
    
    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.connections)
    
    def get_job_connections(self, job_id: str) -> int:
        """Get number of connections for a specific job."""
        return len(self.job_connections.get(job_id, set()))
    
    def get_user_connections(self, user_id: str) -> int:
        """Get number of connections for a specific user."""
        return len(self.user_connections.get(user_id, set()))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics."""
        current_time = time.time()
        
        # Calculate connection durations
        total_duration = 0
        active_connections = []
        
        for connection in self.connections.values():
            duration = current_time - connection.connected_at
            total_duration += duration
            
            active_connections.append({
                "job_id": connection.job_id,
                "user_id": connection.user_id,
                "connected_duration": duration,
                "last_ping": current_time - connection.last_ping
            })
        
        avg_duration = total_duration / len(self.connections) if self.connections else 0
        
        return {
            "total_connections": len(self.connections),
            "job_subscriptions": len(self.job_connections),
            "user_subscriptions": len(self.user_connections),
            "queued_job_messages": len(self.message_queue),
            "average_connection_duration": avg_duration,
            "ping_interval": self.ping_interval,
            "active_connections": active_connections
        }
    
    def clear_message_queue(self, job_id: Optional[str] = None):
        """Clear message queue for a job or all jobs."""
        if job_id:
            if job_id in self.message_queue:
                del self.message_queue[job_id]
                logger.info(f"Cleared message queue for job {job_id}")
        else:
            self.message_queue.clear()
            logger.info("Cleared all message queues")
    
    async def send_system_message(self, message: str, level: str = "info"):
        """Send system message to all connections."""
        await self.broadcast({
            "type": "system_message",
            "message": message,
            "level": level
        })
    
    async def handle_client_message(self, connection_id: str, message: str):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "pong":
                # Handle pong response
                if connection_id in self.connections:
                    self.connections[connection_id].last_ping = time.time()
            
            elif message_type == "subscribe":
                # Handle subscription to job updates
                job_id = data.get("job_id")
                if job_id and connection_id in self.connections:
                    connection = self.connections[connection_id]
                    connection.job_id = job_id
                    
                    if job_id not in self.job_connections:
                        self.job_connections[job_id] = set()
                    self.job_connections[job_id].add(connection_id)
                    
                    await self._send_to_connection(connection_id, {
                        "type": "subscription",
                        "status": "subscribed",
                        "job_id": job_id
                    })
            
            elif message_type == "unsubscribe":
                # Handle unsubscription
                job_id = data.get("job_id")
                if job_id and connection_id in self.connections:
                    if job_id in self.job_connections:
                        self.job_connections[job_id].discard(connection_id)
                    
                    connection = self.connections[connection_id]
                    if connection.job_id == job_id:
                        connection.job_id = None
                    
                    await self._send_to_connection(connection_id, {
                        "type": "subscription",
                        "status": "unsubscribed",
                        "job_id": job_id
                    })
            
            else:
                logger.warning(f"Unknown message type from {connection_id}: {message_type}")
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from {connection_id}: {message}")
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")


# Global WebSocket manager instance
websocket_manager = WebSocketManager() 