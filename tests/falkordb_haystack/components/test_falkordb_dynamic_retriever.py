from typing import List
from unittest import mock

import pytest
from haystack import Document, Pipeline, component

from falkordb_haystack.components.falkordb_retriever import (
    FalkorDBClient,
    FalkorDBClientConfig,
    FalkorDBDynamicDocumentRetriever,
)
from falkordb_haystack.document_stores.falkordb_store import FalkorDBDocumentStore


@pytest.fixture
def client_config(
    falkordb_database: FalkorDBClientConfig,
    doc_store: FalkorDBDocumentStore,
    documents: List[Document],
) -> FalkorDBClientConfig:
    doc_store.write_documents(documents)
    return falkordb_database


@pytest.mark.integration
def test_retrieve_all_documents(client_config: FalkorDBClientConfig):
    retriever = FalkorDBDynamicDocumentRetriever(
        client_config=client_config, doc_node_name="doc", verify_connectivity=True
    )

    result = retriever.run(query="MATCH (doc:Document) RETURN doc")
    documents: List[Document] = result["documents"]

    assert len(documents) == 9


@pytest.mark.integration
def test_compose_docs_from_result(client_config: FalkorDBClientConfig):
    query = """MATCH (doc:Document)
            WITH doc, 10 as score
            RETURN doc.id as id, doc.content as content, score"""

    retriever = FalkorDBDynamicDocumentRetriever(
        client_config=client_config,
        query=query,
        compose_doc_from_result=True,
        verify_connectivity=True,
    )

    result = retriever.run()
    documents: List[Document] = result["documents"]

    assert len(documents) == 9
    for doc in documents:
        assert doc.id
        assert doc.content
        assert doc.score == 10
        assert not doc.meta


@pytest.mark.integration
def test_filter_query(client_config: FalkorDBClientConfig):
    retriever = FalkorDBDynamicDocumentRetriever(
        client_config=client_config, doc_node_name="doc", verify_connectivity=True
    )

    result = retriever.run(
        query="""MATCH (doc:Document)
            WHERE doc.year > 2020 OR doc.year is NULL
            RETURN doc"""
    )
    documents: List[Document] = result["documents"]
    assert len(documents) == 6

    for doc in documents:
        assert doc.meta.get("year", -1) != 2020


@pytest.mark.integration
def test_pipeline_execution(client_config: FalkorDBClientConfig):
    @component
    class YearProvider:
        @component.output_types(year_start=int, year_end=int)
        def run(self, year_start: int, year_end: int):
            return {"year_start": year_start, "year_end": year_end}

    retriever = FalkorDBDynamicDocumentRetriever(
        client_config=client_config,
        runtime_parameters=["year_start", "year_end"],
        doc_node_name="doc",
        verify_connectivity=True,
    )

    query = """MATCH (doc:Document)
            WHERE (doc.year >= $year_start and doc.year <= $year_end) AND doc.month = $month
            RETURN doc LIMIT $num_return"""

    pipeline = Pipeline()
    pipeline.add_component("year_provider", YearProvider())
    pipeline.add_component("retriever", retriever)
    pipeline.connect("year_provider.year_start", "retriever.year_start")
    pipeline.connect("year_provider.year_end", "retriever.year_end")

    result = pipeline.run(
        data={
            "year_provider": {"year_start": 2020, "year_end": 2021},
            "retriever": {
                "query": query,
                "parameters": {
                    "month": "02",
                    "num_return": 2,
                },
            },
        }
    )

    documents = result["retriever"]["documents"]
    assert len(documents) == 2

    for doc in documents:
        assert doc.meta.get("year") == 2021


@pytest.mark.unit
@mock.patch("falkordb_haystack.components.falkordb_retriever.FalkorDBClientConfig", spec=FalkorDBClientConfig)
@mock.patch("falkordb_haystack.components.falkordb_retriever.FalkorDBClient", spec=FalkorDBClient)
def test_document_retriever_to_dict(falkordb_client_mock, client_config_mock):
    falkordb_client = falkordb_client_mock.return_value  # capturing instance created in FalkorDBDynamicDocumentRetriever
    client_config_mock.configure_mock(**{"to_dict.return_value": {"mock": "mock"}})

    retriever = FalkorDBDynamicDocumentRetriever(
        client_config=client_config_mock,
        query="cypher",
        runtime_parameters=["year"],
        doc_node_name="doc",
        compose_doc_from_result=True,
        verify_connectivity=True,
    )

    data = retriever.to_dict()

    assert data == {
        "type": "falkordb_haystack.components.falkordb_retriever.FalkorDBDynamicDocumentRetriever",
        "init_parameters": {
            "query": "cypher",
            "runtime_parameters": ["year"],
            "doc_node_name": "doc",
            "compose_doc_from_result": True,
            "verify_connectivity": True,
            "client_config": {"mock": "mock"},
        },
    }
    falkordb_client.verify_connectivity.assert_called_once()


@pytest.mark.unit
@mock.patch.object(FalkorDBClientConfig, "from_dict")
@mock.patch("falkordb_haystack.components.falkordb_retriever.FalkorDBClient", spec=FalkorDBClient)
def test_document_retriever_from_dict(falkordb_client_mock, from_dict_mock):
    falkordb_client = falkordb_client_mock.return_value  # capturing instance created in FalkorDBDynamicDocumentRetriever
    expected_client_config = mock.Mock(spec=FalkorDBClientConfig)
    from_dict_mock.return_value = expected_client_config

    data = {
        "type": "falkordb_haystack.components.falkordb_retriever.FalkorDBDynamicDocumentRetriever",
        "init_parameters": {
            "query": "cypher",
            "runtime_parameters": ["year"],
            "doc_node_name": "doc",
            "compose_doc_from_result": True,
            "verify_connectivity": True,
            "client_config": {"mock": "mock"},
        },
    }

    retriever = FalkorDBDynamicDocumentRetriever.from_dict(data)

    assert retriever._query == "cypher"
    assert retriever._client_config == expected_client_config
    assert retriever._runtime_parameters == ["year"]
    assert retriever._doc_node_name == "doc"
    assert retriever._compose_doc_from_result is True
    assert retriever._verify_connectivity is True

    falkordb_client.verify_connectivity.assert_called_once()
