import json
import logging
import os
import socket
import time
from typing import Any, Callable, Generator, List

import docker
import numpy as np
import pytest
from haystack import Document
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from falkordb import FalkorDB

from falkordb_haystack.client import FalkorDBClient, FalkorDBClientConfig
from falkordb_haystack.document_stores import FalkorDBDocumentStore

FALKORDB_PORT = 6379
EMBEDDING_DIM = 768

logger = logging.getLogger("conftest")


@pytest.fixture
def documents() -> List[Document]:
    documents = []
    for i in range(3):
        documents.append(
            Document(
                content=f"A Foo Document {i}",
                meta={"name": f"name_{i}", "year": 2020, "month": "01", "numbers": [2, 4]},
                embedding=np.random.rand(EMBEDDING_DIM).astype(np.float32),
            )
        )

        documents.append(
            Document(
                content=f"A Bar Document {i}",
                meta={"name": f"name_{i}", "year": 2021, "month": "02", "numbers": [-2, -4]},
                embedding=np.random.rand(EMBEDDING_DIM).astype(np.float32),
            )
        )

        documents.append(
            Document(
                content=f"Document {i} without embeddings",
                meta={"name": f"name_{i}", "no_embedding": True, "month": "03"},
            )
        )

    return documents


@pytest.fixture
def movie_documents() -> List[Document]:
    current_test_dir = os.path.dirname(__file__)

    with open(os.path.join(current_test_dir, "./samples/movies.json")) as movies_json:
        file_contents = movies_json.read()
        docs_json = json.loads(file_contents)
        documents = [Document.from_dict(doc_json) for doc_json in docs_json]

    return documents


def _get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


def _connection_established(config: FalkorDBClientConfig) -> bool:
    """
    Periodically check FalkorDB database connectivity and return connection status (:const:`True` if has been established)
    """
    timeout = 120
    stop_time = 3
    elapsed_time = 0
    connection_established = False
    while not connection_established and elapsed_time < timeout:
        try:
            client = FalkorDBClient(config)
            client.verify_connectivity()
            client.close_driver()
            connection_established = True
        except Exception as e:
            logger.debug(f"Connection attempt failed: {e}")
            time.sleep(stop_time)
            elapsed_time += stop_time
    return connection_established


@pytest.fixture(scope="module")
def falkordb_database():
    """
    Starts FalkorDB docker container and waits until FalkorDB database is ready.
    Returns FalkorDB client configuration which represents the database in the docker container.
    Container is removed after test suite execution. The `scope` is set to ``module`` to keep only one docker
    container instance for the whole duration of tests execution to speedup the process.
    """
    falkordb_port = _get_free_tcp_port()
    falkordb_version = os.environ.get("FALKORDB_VERSION", "falkordb/falkordb:latest")
    falkordb_container = f"test_falkordb_haystack-{falkordb_port}"
    falkordb_graph = "haystack_test"

    config = FalkorDBClientConfig(
        host="localhost",
        port=falkordb_port,
        graph=falkordb_graph,
    )

    client = docker.from_env()
    container = client.containers.run(
        image=falkordb_version,
        auto_remove=True,
        name=falkordb_container,
        ports={"6379/tcp": ("127.0.0.1", falkordb_port)},
        detach=True,
        remove=True,
    )

    if not _connection_established(config):
        pytest.exit("Could not startup FalkorDB docker container and establish connection with database")

    logger.info(f"Started FalkorDB docker container: {falkordb_container}, image: {falkordb_version}, port: {falkordb_port}")

    yield config

    logger.info(f"Stopping FalkorDB docker container: {falkordb_container}")

    container.stop()


@pytest.fixture
def doc_store_factory(falkordb_database: FalkorDBClientConfig) -> Callable[..., FalkorDBDocumentStore]:
    """
    A factory function to create `FalkorDBDocumentStore`. It depends on the `falkordb_database` fixture (a running
    docker container with falkordb database). Can be used to construct different flavours of `FalkorDBDocumentStore`
    by providing necessary initialization params through keyword arguments (see `doc_store_params`).
    """

    def _doc_store(**doc_store_params) -> FalkorDBDocumentStore:
        return FalkorDBDocumentStore(
            **dict(
                {
                    "host": falkordb_database.host,
                    "port": falkordb_database.port,
                    "graph": falkordb_database.graph,
                    "embedding_dim": EMBEDDING_DIM,
                    "embedding_field": "embedding",
                    "index": "document-embeddings",
                    "node_label": "Document",
                },
                **doc_store_params,
            )
        )

    return _doc_store


@pytest.fixture
def doc_store(doc_store_factory: Callable[..., FalkorDBDocumentStore]) -> Generator[FalkorDBDocumentStore, Any, Any]:
    """
    A default instance of the document store to be used in tests.
    Please notice data (and index) is removed after each test execution to provide a clean state for the next test.
    """
    ds = doc_store_factory()

    yield ds

    # Remove all data from DB to start a new test with clean state
    ds.delete_index()


@pytest.fixture(scope="session")
def text_embedder() -> Callable[[str], List[float]]:
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    text_embedder.warm_up()

    def _text_embedder(text: str) -> List[float]:
        return text_embedder.run(text)["embedding"]

    return _text_embedder


@pytest.fixture(scope="session")
def doc_embedder() -> Callable[[List[Document]], List[Document]]:
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    doc_embedder.warm_up()

    def _doc_embedder(documents: List[Document]) -> List[Document]:
        return doc_embedder.run(documents)["documents"]

    return _doc_embedder


@pytest.fixture
def movie_documents_with_embeddings(
    movie_documents: List[Document],
    doc_embedder: Callable[[List[Document]], List[Document]],
) -> List[Document]:
    documents_copy = [Document.from_dict(doc.to_dict()) for doc in movie_documents]
    return doc_embedder(documents_copy)
