import logging
import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    cast,
)

from haystack import default_from_dict, default_to_dict
from falkordb import FalkorDB

from falkordb_haystack.errors import FalkorDBClientError
from falkordb_haystack.metadata_filter import AST, FalkorDBQueryConverter

logger = logging.getLogger(__name__)

NODE_VAR = "doc"
"""Default variable name used in Cypher queries to match and return Documents, e.g.
`:::cypher match(doc:Document) where doc.id = $id return doc` where `doc` is a variable name."""

DEFAULT_FALKORDB_HOST = "localhost"
"""Default host to connect to FalkorDB instance, e.g. a local DB running in Docker container."""

DEFAULT_FALKORDB_PORT = 6379
"""Default port to connect to FalkorDB instance."""

DEFAULT_FALKORDB_GRAPH = "haystack"
"""Default FalkorDB graph name to use if not provided."""

DEFAULT_FALKORDB_USERNAME = None
"""Default FalkorDB username to be used for authentication with FalkorDB."""

DEFAULT_FALKORDB_PASSWORD = None
"""Default FalkorDB password to be used for authentication with FalkorDB."""

FalkorDBRecord = Dict[str, Any]
"""Type alias for data items returned from FalkorDB queries"""

SimilarityFunction = Literal["cosine", "euclidean"]


@dataclass
class VectorStoreIndexInfo:
    """FalkorDB vector index information retrieved from the database.

    See [Create and configure vector indexes](https://docs.falkordb.com/cypher/indexing.html)
    documentation to learn more about data representing index configuration.

    Attributes:
        index_name: The name of the index.
        node_label: Name of FalkorDB node which contains embeddings which are indexed.
        property_key: Name of the property of the node which contains vectors.
        dimensions: Dimension of embedding vector.
        similarity_function: Configured vector similarity function.
    """

    index_name: str
    node_label: str
    property_key: str
    dimensions: int
    similarity_function: str


@dataclass
class FalkorDBClientConfig:
    """
    Provides configuration options to communicate with FalkorDB database.

    FalkorDB uses Redis protocol, so connection is made via host/port rather than URL.

    Attributes:
        host: Database host address (default: localhost)
        port: Database port (default: 6379)
        graph: Graph name to use (default: haystack)
        username: Username to authenticate with the database (optional)
        password: Password credential for the given username (optional)
        use_env: If `True` the following attributes will be assigned from respective environment variables:
            ```py
            >>> host = os.getenv("FALKORDB_HOST")
            >>> port = os.getenv("FALKORDB_PORT")
            >>> graph = os.getenv("FALKORDB_GRAPH")
            >>> username = os.getenv("FALKORDB_USERNAME")
            >>> password = os.getenv("FALKORDB_PASSWORD")
            ```
    """

    host: str = field(default=DEFAULT_FALKORDB_HOST)
    port: int = field(default=DEFAULT_FALKORDB_PORT)
    graph: str = field(default=DEFAULT_FALKORDB_GRAPH)
    username: Optional[str] = field(default=DEFAULT_FALKORDB_USERNAME)
    password: Optional[str] = field(default=DEFAULT_FALKORDB_PASSWORD)
    use_env: Optional[bool] = field(default=False)

    def __post_init__(self):
        if self.use_env:
            self.host = os.getenv("FALKORDB_HOST", self.host)
            port_env = os.getenv("FALKORDB_PORT")
            if port_env:
                self.port = int(port_env)
            self.graph = os.getenv("FALKORDB_GRAPH", self.graph)
            self.username = os.getenv("FALKORDB_USERNAME", self.username)
            self.password = os.getenv("FALKORDB_PASSWORD", self.password)

        if not self.host:
            raise ValueError("The `host` attribute is mandatory to connect to database.")
        if not self.graph:
            raise ValueError("The `graph` attribute is mandatory to select a graph.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes client configuration to a dictionary.
        """
        data = default_to_dict(
            self,
            host=self.host,
            port=self.port,
            graph=self.graph,
            username=self.username,
            password=self.password,
            use_env=self.use_env,
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FalkorDBClientConfig":
        """
        Deserializes client configuration from a dictionary.
        """
        return default_from_dict(cls, data)


class FalkorDBClient:
    """
    FalkorDB Python Driver wrapper to run low level database transactions using Cypher queries. It abstracts away FalkorDB
    related details from `FalkorDBDocumentStore` so that database related interactions are encapsulated in a single
    place.

    `FalkorDBClient` can be created with a number of configuration options represented by the `FalkorDBClientConfig` data
    class. The configuration applied when connecting to a database or running transactions.

    Attributes:
        _config: FalkorDB configuration options.
        _db: An instance of FalkorDB client connection.
        _graph: The graph object for executing queries.
        _filter_converter: Instance of `FalkorDBQueryConverter` which converts parsed Metadata filters to Cypher
            queries.
    """

    def __init__(self, config: FalkorDBClientConfig):
        self._config = config

        if not config.host:
            raise ValueError("`FalkorDBClientConfig.host` is mandatory attribute when trying to connect to FalkorDB database.")

        # Create FalkorDB connection
        kwargs = {"host": config.host, "port": config.port}
        if config.username and config.password:
            kwargs["username"] = config.username
            kwargs["password"] = config.password

        self._db = FalkorDB(**kwargs)
        self._graph = self._db.select_graph(config.graph)
        self._filter_converter = FalkorDBQueryConverter(NODE_VAR)

    def delete_nodes(self, node_label: str, filter_ast: Optional[AST] = None) -> None:
        """
        Deletes nodes with with given label and filters using [DELETE](https://docs.falkordb.com/commands/)
            Cypher clause.

        Args:
            node_label: The name of the label to delete (e.g. ``"Document"``)
            filter_ast: Metadata filters to delete only specific nodes which match filtering conditions.
        """
        where_clause, where_params = self._where_clause(filter_ast)
        query = f"""
            MATCH ({NODE_VAR}:`{node_label}`)
            {where_clause}
            DELETE {NODE_VAR}
        """
        self._graph.query(query, where_params)

    def create_index(
        self,
        index_name: str,
        label: str,
        property_key: str,
        dimension: int,
        similarity_function: SimilarityFunction,
    ) -> None:
        """
        Creates a new vector index in database for a given node label and vector specific attributes.
        See documentation for vector indexes in FalkorDB.

        Args:
            index_name: The unique name of the index.
            label: The node label to be indexed (e.g. ``"Document"``).
            property_key: The property key of a node which contains embedding values.
            dimension: Vector embedding dimension (must be between 1 and 2048 inclusively).
            similarity_function: case-insensitive values for the vector similarity function:
                ``cosine`` or ``euclidean``.
        """
        query = f"""
            CREATE VECTOR INDEX FOR (n:{label}) ON (n.{property_key})
            OPTIONS {{dimension: {dimension}, similarityFunction: '{similarity_function}'}}
        """
        self._graph.query(query)

    def retrieve_vector_index(
        self,
        index_name: str,
        node_label: str,
        property_key: str,
    ) -> Optional[VectorStoreIndexInfo]:
        """
        Retrieves information about existing vector index.

        Args:
            index_name: The name of the vector index to retrieve.
            node_label: The label of the node configured as part of vector index setup.
            property_key: The property key configured as part of vector index setup.

        Returns:
            Data retrieved from the query or `None` if index was not found.
        """
        # FalkorDB doesn't have a SHOW INDEXES equivalent in the same way as Neo4j
        # We'll need to check if the index exists by querying it
        # For now, return None - this needs proper implementation based on FalkorDB's index introspection
        logger.warning("Vector index retrieval not fully implemented for FalkorDB yet")
        return None

    def create_index_if_missing(
        self,
        index_name: str,
        label: str,
        property_key: str,
        dimension: int,
        similarity_function: SimilarityFunction,
    ):
        """
        Creates a vector index in case it does not exist in database.
        """
        existing_index = self.retrieve_vector_index(index_name, label, property_key)

        if not existing_index:
            logger.debug("Creating a new index(%s) as it is not present in the configured FalkorDB database", index_name)
            try:
                self.create_index(index_name, label, property_key, dimension, similarity_function)
            except Exception as e:
                # Index might already exist, log and continue
                logger.debug("Index creation attempt resulted in: %s", str(e))

    def delete_index(self, index_name: str) -> None:
        """
        Removes index from FalkorDB database.

        Args:
            index_name: The name of the index to delete.
        """
        query = f"DROP INDEX {index_name}"
        self._graph.query(query)

    def update_embedding(self, node_label: str, embedding_field: str, records: List[Dict[str, Any]]) -> None:
        """
        Updates embedding on a number of ``Document`` nodes.

        Args:
            node_label: A node label to match (e.g. ``"Document"``).
            embedding_field: The name of the embedding field which stores embeddings.
            records: A list dictionary objects following the structure:
                ```python
                    [{
                        "id": "doc_id1", # id of the Document (node) to update
                        embedding_field: [0.8, 0.9, ...] # Embedding vector
                    }]
                ```
        """
        for record in records:
            doc_id = record["id"]
            embedding = record.get(embedding_field)
            if embedding:
                # Convert embedding to vecf32 format
                vec_str = "vecf32([" + ",".join(str(v) for v in embedding) + "])"
                query = f"""
                    MATCH ({NODE_VAR}:`{node_label}` {{id: $id}})
                    SET {NODE_VAR}.{embedding_field} = {vec_str}
                    RETURN {NODE_VAR}
                """
                self._graph.query(query, {"id": doc_id})

    def merge_nodes(self, node_label: str, embedding_field: str, records: List[FalkorDBRecord]) -> Any:
        """
        Creates or updates a node in FalkorDB representing a Document with all properties. Nodes are matched by "id",
        if not found a new node will be created.

        Args:
            node_label: The label of the node to match (e.g. "Document").
            embedding_field: The name of the embedding field which stores embeddings.
            records: A list of [Documents](https://docs.haystack.deepset.ai/reference/primitives-api#document)
                converted to dictionaries, with ``meta`` attributes included.
        """
        for record in records:
            doc_id = record["id"]
            # Separate embedding from other properties
            embedding = record.pop(embedding_field, None)
            
            # Build SET clause for non-embedding properties
            set_props = []
            params = {"id": doc_id}
            for key, value in record.items():
                if key != "id" and value is not None:
                    param_name = f"prop_{key}"
                    set_props.append(f"{NODE_VAR}.{key} = ${param_name}")
                    params[param_name] = value
            
            set_clause = ", ".join(set_props) if set_props else ""
            
            # Create or update node
            if set_clause:
                query = f"""
                    MERGE ({NODE_VAR}:`{node_label}` {{id: $id}})
                    SET {set_clause}
                    RETURN {NODE_VAR}
                """
            else:
                query = f"""
                    MERGE ({NODE_VAR}:`{node_label}` {{id: $id}})
                    RETURN {NODE_VAR}
                """
            
            self._graph.query(query, params)
            
            # Update embedding separately if present
            if embedding is not None:
                vec_str = "vecf32([" + ",".join(str(v) for v in embedding) + "])"
                embed_query = f"""
                    MATCH ({NODE_VAR}:`{node_label}` {{id: $id}})
                    SET {NODE_VAR}.{embedding_field} = {vec_str}
                    RETURN {NODE_VAR}
                """
                self._graph.query(embed_query, {"id": doc_id})
        
        # Return a dummy summary object
        return type('Summary', (), {'counters': type('Counters', (), {})()})()

    def count_nodes(self, node_label: str, filter_ast: Optional[AST] = None) -> int:
        """
        Counts number of nodes matching given label and optional filters.

        Args:
            node_label: The label of the node to match (e.g. ``"Document"``).
            filter_ast: The filter syntax tree (parsed metadata filter) to narrow down counted results.

        Returns:
            Number of found nodes.
        """
        where_clause, where_params = self._where_clause(filter_ast)
        query = f"""
            MATCH ({NODE_VAR}:`{node_label}`)
            {where_clause}
            RETURN count(*) as count
        """
        result = self._graph.query(query, where_params)
        if result.result_set:
            return result.result_set[0][0]
        return 0

    def find_nodes(
        self,
        node_label: str,
        filter_ast: Optional[AST] = None,
        skip_properties: Optional[List[str]] = None,
        fetch_size: int = 1000,
    ) -> Generator[FalkorDBRecord, None, None]:
        """
        Search for nodes matching a given label and metadata filters.

        Args:
            node_label: The label of the nodes to match (e.g. ``"Document"``).
            filter_ast: The filter syntax tree (parsed metadata filter) for search.
            skip_properties: Properties we would like not to return as part of data payload.
            fetch_size: Controls how many records are fetched at once from the database.

        Returns:
            Found records matching search criteria.
        """
        where_clause, where_params = self._where_clause(filter_ast)
        query = f"""
            MATCH ({NODE_VAR}:`{node_label}`)
            {where_clause}
            RETURN {NODE_VAR}
        """

        for record in self.query_nodes(query=query, parameters=where_params, fetch_size=fetch_size):
            yield record

    def query_nodes(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        fetch_size: int = 1000,
    ) -> Generator[FalkorDBRecord, None, None]:
        """
        Runs a given Cypher `query`.

        Args:
            query: Cypher query to run in FalkorDB.
            parameters: Query parameters which can be used as placeholders in the `query`.
            fetch_size: Controls how many records are fetched at once from the database.

        Returns:
            Records containing data specified in ``RETURN`` Cypher query statement.
        """
        result = self._graph.query(query, parameters or {})
        
        if result.result_set:
            for row in result.result_set:
                # Extract node data from result
                if row:
                    node_data = row[0] if isinstance(row[0], dict) else self._node_to_dict(row[0])
                    yield node_data

    def query_embeddings(
        self,
        index: str,
        top_k: int,
        embedding: List[float],
        filter_ast: Optional[AST] = None,
        skip_properties: Optional[List[str]] = None,
        vector_top_k: Optional[int] = None,
    ) -> List[FalkorDBRecord]:
        """
        Query a vector index and apply filtering using `WHERE` clause on results returned by vector search.

        Args:
            index: Refers to the unique name of the vector index to query.
            top_k: Number of results to return from vector search.
            embedding: The query vector in which to search for the neighborhood.
            filter_ast: Additional filters translated into `WHERE` Cypher clause.
            skip_properties: Properties we would like **not** to return as part of data payload.
            vector_top_k: If provided `vector_top_k` is used instead of `top_k`.

        Returns:
            An ordered by score `top_k` nodes found in vector search.
        """
        if vector_top_k and vector_top_k < top_k:
            logger.warning(
                "Make sure 'vector_top_k'(=%s) is greater than 'top_k'(=%s) parameter. Using 'top_k' instead",
                vector_top_k,
                top_k,
            )
            vector_top_k = top_k

        where_clause, where_params = self._where_clause(filter_ast)
        
        # Convert embedding to vecf32 format
        vec_str = "vecf32([" + ",".join(str(v) for v in embedding) + "])"
        
        # FalkorDB vector search query
        query = f"""
            CALL db.idx.vector.queryNodes($index_name, $property_key, $vector_top_k, {vec_str})
            YIELD node as {NODE_VAR}, score
            MATCH ({NODE_VAR}) {where_clause}
            RETURN {NODE_VAR}, score
            ORDER BY score DESC LIMIT $top_k
        """
        
        params = {
            "index_name": index,
            "property_key": "embedding",  # This should be configurable
            "top_k": top_k,
            "vector_top_k": vector_top_k or top_k,
            **where_params,
        }

        result = self._graph.query(query, params)
        
        records = []
        if result.result_set:
            for row in result.result_set:
                node_data = row[0] if isinstance(row[0], dict) else self._node_to_dict(row[0])
                score = row[1]
                node_data["score"] = score
                records.append(node_data)
        
        return records

    def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Runs an arbitrary write Cypher query with parameters.

        Args:
            query: Cypher query to run in FalkorDB.
            parameters: Query parameters which can be used as placeholders in the `query`.

        Returns:
            A tuple consisting of execution result summary and data records if any.
        """
        result = self._graph.query(query, parameters or {})
        
        records = []
        if result.result_set:
            for row in result.result_set:
                if row:
                    record_dict = {}
                    # Try to extract data from row
                    for i, item in enumerate(row):
                        if isinstance(item, dict):
                            record_dict.update(item)
                        else:
                            record_dict[f"col_{i}"] = item
                    records.append(record_dict)
        
        return (result, records)

    def execute_read(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Runs an arbitrary "read" Cypher query with parameters.

        Args:
            query: Cypher query to run in FalkorDB.
            parameters: Query parameters which can be used as placeholders in the `query`.

        Returns:
            A tuple consisting of execution result summary and data records if any.
        """
        # In FalkorDB, there's no distinction between read and write at the API level
        # Use ro_query for read-only operations
        result = self._graph.ro_query(query, parameters or {})
        
        records = []
        if result.result_set:
            for row in result.result_set:
                if row:
                    record_dict = {}
                    for i, item in enumerate(row):
                        if isinstance(item, dict):
                            record_dict.update(item)
                        else:
                            record_dict[f"col_{i}"] = item
                    records.append(record_dict)
        
        return (result, records)

    def update_node(self, node_label: str, doc_id: str, data: Dict[str, Any]) -> Optional[FalkorDBRecord]:
        """
        Updates a given node matched by the given id (`doc_id`).

        Args:
            node_label: A node label to match (e.g. "Document").
            doc_id: Node id to match.
            data: A dictionary of data which will be set as node's properties.

        Returns:
            Updated FalkorDB record data.
        """
        # Build SET clause
        set_props = []
        params = {"doc_id": doc_id}
        for key, value in data.items():
            param_name = f"prop_{key}"
            set_props.append(f"{NODE_VAR}.{key} = ${param_name}")
            params[param_name] = value
        
        if not set_props:
            return None
        
        set_clause = ", ".join(set_props)
        query = f"""
            MATCH ({NODE_VAR}:`{node_label}` {{id: $doc_id}})
            SET {set_clause}
            RETURN {NODE_VAR}
        """
        
        result = self._graph.query(query, params)
        if result.result_set and result.result_set[0]:
            return self._node_to_dict(result.result_set[0][0])
        return None

    def verify_connectivity(self):
        """
        Verifies connection to FalkorDB database as per configuration and auth credentials provided.

        Raises:
            FalkorDBClientError: In case connection could not be established.
        """
        try:
            # Try a simple query to verify connectivity
            self._graph.query("RETURN 1")
        except Exception as err:
            raise FalkorDBClientError(
                "Could not connect to FalkorDB database. Please ensure that the host, port and provided credentials are correct"
            ) from err

    def close_driver(self) -> None:
        """Close the FalkorDB connection."""
        logger.debug("Closing FalkorDB client connection")
        if hasattr(self._db, 'close'):
            self._db.close()

    def _where_clause(self, filter_ast: Optional[AST]) -> Tuple[str, Dict[str, Any]]:
        """
        Converts a given filter syntax tree `filter_ast` into a Cypher query to build ``WHERE`` filter clause.

        Args:
            filter_ast: Filters AST to be converted into Cypher query.

        Returns:
            ``WHERE`` filter clause and parameters used in the clause.
        """
        if filter_ast:
            query, params = self._filter_converter.convert(filter_ast)
            return f"WHERE {query}", params

        # empty query and no parameters
        return ("", {})

    def _node_to_dict(self, node: Any) -> Dict[str, Any]:
        """
        Convert a FalkorDB node object to a dictionary.

        Args:
            node: A FalkorDB node object.

        Returns:
            Dictionary representation of the node.
        """
        if isinstance(node, dict):
            return node
        
        # FalkorDB nodes have properties attribute
        if hasattr(node, 'properties'):
            return dict(node.properties)
        
        # Fallback: try to convert to dict
        try:
            return dict(node)
        except:
            logger.warning("Could not convert node to dict: %s", type(node))
            return {}
