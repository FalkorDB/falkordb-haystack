from typing import List
from unittest import mock

import pytest
from haystack import Document

from falkordb_haystack.client.falkordb_client import FalkorDBClientConfig
from falkordb_haystack.components.falkordb_query_reader import FalkorDBQueryReader
from falkordb_haystack.components.falkordb_retriever import FalkorDBClient
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
def test_query_reader(client_config: FalkorDBClientConfig):
    query = "MATCH (doc:`Document`) WHERE doc.year=$year RETURN doc.name as name, doc.year as year"
    reader = FalkorDBQueryReader(
        client_config=client_config,
        query=query,
        verify_connectivity=True,
        runtime_parameters=["year"],
    )

    result = reader.run(year=2020)

    assert result["records"] == [
        {"name": "name_0", "year": 2020},
        {"name": "name_1", "year": 2020},
        {"name": "name_2", "year": 2020},
    ]


@pytest.mark.integration
def test_query_reader_single_record(client_config: FalkorDBClientConfig):
    reader = FalkorDBQueryReader(client_config=client_config, verify_connectivity=True)

    result = reader.run(
        query=("MATCH (doc:`Document` {name: $name}) RETURN doc.name as name, doc.year as year"),
        parameters={"name": "name_1"},
    )

    assert result["first_record"] == {"name": "name_1", "year": 2020}


@pytest.mark.integration
def test_query_reader_error_result(client_config: FalkorDBClientConfig):
    reader = FalkorDBQueryReader(client_config=client_config, raise_on_failure=False)

    result = reader.run(
        query=("MATCH (doc:`Document` {name: $name}) RETURN_ doc.name as name, doc.year as year"),
        parameters={"name": "name_1"},
    )

    assert "Invalid input 'RETURN_'" in result["error_message"]


@pytest.mark.integration
def test_query_reader_raises_error(client_config: FalkorDBClientConfig):
    reader = FalkorDBQueryReader(client_config=client_config, raise_on_failure=True)

    with pytest.raises(Exception):  # noqa: B017
        reader.run(
            query=("MATCH (doc:`Document` {name: $name}) RETURN_ doc.name as name, doc.year as year"),
            parameters={"name": "name_1"},
        )


@pytest.mark.unit
@mock.patch("falkordb_haystack.components.falkordb_query_reader.FalkorDBClientConfig", spec=FalkorDBClientConfig)
@mock.patch("falkordb_haystack.components.falkordb_query_reader.FalkorDBClient", spec=FalkorDBClient)
def test_falkordb_query_reader_to_dict(falkordb_client_mock, client_config_mock):
    falkordb_client = falkordb_client_mock.return_value  # capturing instance created in FalkorDBQueryReader
    client_config_mock.configure_mock(**{"to_dict.return_value": {"mock": "mock"}})

    reader = FalkorDBQueryReader(
        client_config=client_config_mock,
        query="cypher",
        runtime_parameters=["year"],
        verify_connectivity=True,
        raise_on_failure=False,
    )

    data = reader.to_dict()

    assert data == {
        "type": "falkordb_haystack.components.falkordb_query_reader.FalkorDBQueryReader",
        "init_parameters": {
            "query": "cypher",
            "runtime_parameters": ["year"],
            "verify_connectivity": True,
            "raise_on_failure": False,
            "client_config": {"mock": "mock"},
        },
    }
    falkordb_client.verify_connectivity.assert_called_once()


@pytest.mark.unit
@mock.patch.object(FalkorDBClientConfig, "from_dict")
@mock.patch("falkordb_haystack.components.falkordb_query_reader.FalkorDBClient", spec=FalkorDBClient)
def test_falkordb_query_reader_from_dict(falkordb_client_mock, from_dict_mock):
    falkordb_client = falkordb_client_mock.return_value  # capturing instance created in FalkorDBQueryReader
    expected_client_config = mock.Mock(spec=FalkorDBClientConfig)
    from_dict_mock.return_value = expected_client_config

    data = {
        "type": "falkordb_haystack.components.falkordb_query_reader.FalkorDBQueryReader",
        "init_parameters": {
            "query": "cypher",
            "runtime_parameters": ["year"],
            "verify_connectivity": True,
            "raise_on_failure": False,
            "client_config": {"mock": "mock"},
        },
    }

    reader = FalkorDBQueryReader.from_dict(data)

    assert reader._query == "cypher"
    assert reader._client_config == expected_client_config
    assert reader._runtime_parameters == ["year"]
    assert reader._verify_connectivity is True
    assert reader._raise_on_failure is False

    falkordb_client.verify_connectivity.assert_called_once()
