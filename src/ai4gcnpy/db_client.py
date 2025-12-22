"""
A well-structured Neo4j graph database client for GCN circular data ingestion.
Supports safe deletion (only deletes nodes created by this program) and batch operations.
"""
from neo4j import Driver, Session, GraphDatabase, Auth, Transaction, Result
from neo4j_graphrag.schema import get_schema
from contextlib import contextmanager
from typing import Optional, Dict, Any, Generator
from datetime import datetime, date
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
        driver_config: Dict[str, Any] = {}  
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

        # Build auth tuple safely
        auth = Auth(
            "basic",
            username or os.getenv("NEO4J_USERNAME", "neo4j"),
            password or os.getenv("NEO4J_PASSWORD", "neo4j"),
        )

        try:
            self._driver: Driver = GraphDatabase.driver(
                self.url, auth=auth, **driver_config
            )
            # Verify connectivity 
            self._driver.verify_connectivity()
            logger.debug(f"Successfully connected to Neo4j at '{self.url}'")
        except Exception as e:
            raise ValueError(f"Failed to connect to Neo4j: {e}") from e

    @contextmanager
    def session(self, database: Optional[str] = None) -> Generator[Session, None, None]:
        """
        Context manager for Neo4j session with optional database selection.

        Args:
            database: Optional name of the target Neo4j database. If not provided, uses the default database.

        Yields:
            A Neo4j Session object scoped to the specified database.
        """
        session_kwargs: Dict[str, Any] = {"database": database} if database else {}

        with self._driver.session(**session_kwargs) as session:
            logger.debug(f"Opened Neo4j session on database '{database or '<default>'}'")
            yield session

    @contextmanager
    def transaction(self, database: Optional[str] = None) -> Generator[Transaction, None, None]:
        """
        Context manager for Neo4j transaction with optional database selection.

        Args:
            database: Optional name of the target Neo4j database. If not provided, uses the default database.

        Yields:
            A Neo4j Transaction object scoped to the specified database.
        """
        session_kwargs: Dict[str, Any] = {"database": database} if database else {}
        db_name = database or "<default>"

        with self._driver.session(**session_kwargs) as session:
            logger.debug(f"Opened Neo4j session on database '{db_name}'")
            with session.begin_transaction() as tx:
                logger.debug(f"Started Neo4j transaction on database '{db_name}'")
                yield tx
            logger.debug(f"CloseOperation: Transaction ended on database: '{db_name}'")
        logger.debug(f"CloseOperation: Session closed for database: '{db_name}'")

    def get_schema(self, database: Optional[str] = None) -> str:
        try:
            schema_str = get_schema(self._driver, database=database, is_enhanced=True)
            logger.debug("Retrieved schema via neo4j_graphrag.schema.get_schema")
            return schema_str
        except Exception as e:
            logger.error(f"Failed to retrieve schema using neo4j_graphrag: {e}")
            return ""

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        if hasattr(self, '_driver') and self._driver:
            self._driver.close()
            logger.debug("Neo4j driver closed.")

    def delete_all(self, created_at: str, create_by: str = "AI4GCNpy", database: Optional[str] = None) -> None:
        """
        Delete nodes and relationships created by this client on a specific date.

        Args:
            created_at: Date string in 'YYYY-MM-DD' format.
        """
        target_date = date.fromisoformat(created_at)

        with self.session() as session:
            # 1. Delete relationships first
            rel_query = """
            MATCH ()-[r]-()
            WHERE r.ingestedBy = $create_by AND date(r.ingestedAt) = $created_at
            DELETE r
            RETURN count(r) AS rels
            """
            rel_record = session.run(rel_query, create_by=create_by, created_at=target_date).single()
            rels_deleted = rel_record["rels"] if rel_record else 0

            # 2. Delete nodes
            node_query = """
            MATCH (n)
            WHERE n.ingestedBy = $create_by AND date(n.ingestedAt) = $created_at
            DETACH DELETE n
            RETURN count(n) AS nodes
            """
            node_record = session.run(node_query, create_by=create_by, created_at=target_date).single()
            nodes_deleted = node_record["nodes"] if node_record else 0

        logger.info(
            f"Deleted {nodes_deleted} nodes and {rels_deleted} relationships created by {create_by} on {created_at}"
        )

