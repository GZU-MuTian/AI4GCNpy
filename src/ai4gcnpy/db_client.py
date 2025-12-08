"""
A well-structured Neo4j graph database client for GCN circular data ingestion.
Supports safe deletion (only deletes nodes created by this program) and batch operations.
"""
from neo4j import Driver, GraphDatabase, Auth
from neo4j_graphrag.schema import get_schema
from contextlib import contextmanager
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import logging
import os

logger = logging.getLogger(__name__)
load_dotenv()


class GCNGraphDB:
    """
    A class to handle the connection to the Neo4j database and perform GCN-specific operations.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        driver_config: Optional[Dict[str, Any]] = None  
    ):
        """
        Initialize Neo4j driver with flexible configuration.

        Args:
            url: Neo4j Bolt URI (default from NEO4J_URI env var).
            username: Database username (default from NEO4J_USERNAME).
            password: Database password (default from NEO4J_PASSWORD).
            driver_config: Additional config passed to GraphDatabase.driver().
        """
        self.url = url or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.driver_config = driver_config or {}
        self._create_by = "ai4gcnpy"

        # Build auth tuple safely
        auth = Auth(
            "basic",
            username or os.getenv("NEO4J_USERNAME", "neo4j"),
            password or os.getenv("NEO4J_PASSWORD", "neo4j"),
        )

        try:
            self._driver: Driver = GraphDatabase.driver(
                self.url, auth=auth, **self.driver_config
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at '{self.url}'")
        except Exception as e:
            raise ValueError(f"Failed to connect to Neo4j: {e}") from e

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        if hasattr(self, '_driver') and self._driver:
            self._driver.close()
            logger.info("Neo4j driver closed.")

    def get_schema(self) -> str:
        try:
            schema_str = get_schema(self._driver)
            logger.debug("Retrieved schema via neo4j_graphrag.schema.get_schema")
            return schema_str
        except Exception as e:
            logger.error(f"Failed to retrieve schema using neo4j_graphrag: {e}")
            return ""

    def delete_all(self, created_at: str) -> None:
        """
        Delete nodes and relationships created by this client on a specific date.

        Args:
            created_at: Date string in 'YYYY-MM-DD' format.
        """
        with self._driver.session() as session:
            # 1. Delete relationships first
            rel_query = """
            MATCH ()-[r]-()
            WHERE r.create_by = $create_by AND r.created_at = $date
            DELETE r
            RETURN count(r) AS rels
            """
            rel_result = session.run(rel_query, create_by=self._create_by, date=created_at)
            # rels_deleted = rel_result.single()["rels"]

            # 2. Delete nodes
            node_query = """
            MATCH (n)
            WHERE n.create_by = $create_by AND n.created_at = $date
            DETACH DELETE n
            RETURN count(n) AS nodes
            """
            node_result = session.run(node_query, create_by=self._create_by, date=created_at)
            # nodes_deleted = node_result.single()["nodes"]

        # logger.info(f"Deleted {nodes_deleted} nodes and {rels_deleted} relationships created by {self._create_by} on {created_at}")


    # WRITE OPERATIONS

