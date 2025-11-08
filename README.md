<h1 align="center">falkordb-haystack</h1>

<p align="center">A <a href="https://docs.haystack.deepset.ai/docs/document_store"><i>Haystack</i></a> Document Store for <a href="https://falkordb.com/"><i>FalkorDB</i></a>.</p>

<p align="center">
  <a href="https://github.com/prosto/falkordb-haystack/actions?query=workflow%3Aci">
    <img alt="ci" src="https://github.com/prosto/falkordb-haystack/workflows/ci/badge.svg" />
  </a>
  <a href="https://prosto.github.io/falkordb-haystack/">
    <img alt="documentation" src="https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat" />
  </a>
  <a href="https://pypi.org/project/falkordb-haystack/">
    <img alt="pypi version" src="https://img.shields.io/pypi/v/falkordb-haystack.svg" />
  </a>
  <a href="https://img.shields.io/pypi/pyversions/falkordb-haystack.svg">
    <img alt="python version" src="https://img.shields.io/pypi/pyversions/falkordb-haystack.svg" />
  </a>
  <a href="https://pypi.org/project/haystack-ai/">
    <img alt="haystack version" src="https://img.shields.io/pypi/v/haystack-ai.svg?label=haystack" />
  </a>
</p>

---

**Table of Contents**

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Running FalkorDB](#running-falkordb)
  - [Document Store](#document-store)
  - [Indexing documents](#indexing-documents)
  - [Retrieving documents](#retrieving-documents)
  - [Retrieving documents using Cypher](#retrieving-documents-using-cypher)
  - [More examples](#more-examples)
- [License](#license)

## Overview

An integration of [FalkorDB](https://falkordb.com/) graph database with [Haystack v2.0](https://docs.haystack.deepset.ai/v2.0/docs/intro)
by [deepset](https://www.deepset.ai). In FalkorDB [Vector search index](https://falkordb.com/docs/cypher-manual/current/indexes-for-vector-search/)
is being used for storing document embeddings and dense retrievals.

The library allows using FalkorDB as a [DocumentStore](https://docs.haystack.deepset.ai/v2.0/docs/document-store), and implements the required [Protocol](https://docs.haystack.deepset.ai/v2.0/docs/document-store#documentstore-protocol) methods. You can start working with the implementation by importing it from `neo4_haystack` package:

```python
from falkordb_haystack import FalkorDBDocumentStore
```

In addition to the `FalkorDBDocumentStore` the library includes the following haystack components which can be used in a pipeline:

- [FalkorDBEmbeddingRetriever](https://prosto.github.io/falkordb-haystack/reference/falkordb_retriever/#falkordb_haystack.components.falkordb_retriever.FalkorDBEmbeddingRetriever) - is a typical [retriever component](https://docs.haystack.deepset.ai/v2.0/docs/retrievers) which can be used to query vector store index and find related Documents. The component uses `FalkorDBDocumentStore` to query embeddings.
- [FalkorDBDynamicDocumentRetriever](https://prosto.github.io/falkordb-haystack/reference/falkordb_retriever/#falkordb_haystack.components.falkordb_retriever.FalkorDBDynamicDocumentRetriever) is also a retriever component in a sense that it can be used to query Documents in FalkorDB. However it is decoupled from `FalkorDBDocumentStore` and allows to run arbitrary [Cypher query](https://falkordb.com/docs/cypher-manual/current/queries/) to extract documents. Practically it is possible to query FalkorDB same way `FalkorDBDocumentStore` does, including vector search.
- [FalkorDBQueryReader](https://prosto.github.io/falkordb-haystack/reference/falkordb_query_reader/#falkordb_haystack.components.falkordb_query_reader.FalkorDBQueryReader) - is a component which gives flexible way to read data from FalkorDB by running custom Cypher query along with query parameters. You could use such queries to read data from FalkorDB to enhance your RAG pipelines. For example prompting LLM to produce Cypher query based on given context (Text to Cypher) and use `FalkorDBQueryReader` to run the
  query and extract results. [OutputAdapter](https://docs.haystack.deepset.ai/docs/outputadapter) component might
  become handy in such scenarios - it can be used to handle outputs from `FalkorDBQueryReader`.
- [FalkorDBQueryWriter](https://prosto.github.io/falkordb-haystack/reference/falkordb_query_writer/#falkordb_haystack.components.falkordb_query_writer.FalkorDBQueryWriter) - this component gives flexible way to write data to FalkorDB by running arbitrary Cypher query along with parameters. Query parameters can be pipeline inputs or outputs from connected components. You could use such queries to write Documents with additional graph nodes for a more complex RAG scenarios. The difference between [DocumentWriter](https://docs.haystack.deepset.ai/docs/documentwriter) and `FalkorDBQueryWriter` is that the latter can write any data to FalkorDB, not just Documents.

The `falkordb-haystack` library uses [Python Driver](https://falkordb.com/docs/api/python-driver/current/api.html#api-documentation) and
[Cypher Queries](https://falkordb.com/docs/cypher-manual/current/introduction/) to interact with FalkorDB database and hide all complexities under the hood.

`FalkorDBDocumentStore` will store Documents as Graph nodes in FalkorDB. Embeddings are stored as part of the node, but indexing and querying of vector embeddings using ANN is managed by a dedicated [Vector Index](https://falkordb.com/docs/cypher-manual/current/indexes-for-vector-search/).

```text
                                   +-----------------------------+
                                   |       FalkorDB Database        |
                                   +-----------------------------+
                                   |                             |
                                   |      +----------------+     |
                                   |      |    Document    |     |
                write_documents    |      +----------------+     |
          +------------------------+----->|   properties   |     |
          |                        |      |                |     |
+---------+----------+             |      |   embedding    |     |
|                    |             |      +--------+-------+     |
| FalkorDBDocumentStore |             |               |             |
|                    |             |               |index/query  |
+---------+----------+             |               |             |
          |                        |      +--------+--------+    |
          |                        |      |  Vector Index   |    |
          +----------------------->|      |                 |    |
               query_embeddings    |      | (for embedding) |    |
                                   |      +-----------------+    |
                                   |                             |
                                   +-----------------------------+
```

In the above diagram:

- `Document` is a FalkorDB node (with "Document" label)
- `properties` are Document [attributes](https://docs.haystack.deepset.ai/v2.0/docs/data-classes#document) stored as part of the node. **In current implementation `meta` attributes are stored on the same level as the rest of Document fields.**
- `embedding` is also a property of the Document node (just shown separately in the diagram for clarity) which is a vector of type `LIST[FLOAT]`.
- `Vector Index` is where embeddings are getting indexed by FalkorDB as soon as those are updated in Document nodes.

`FalkorDBDocumentStore` by default creates a vector index if it does not exist. Before writing documents you should make sure Documents are embedded by one of the provided [embedders](https://docs.haystack.deepset.ai/v2.0/docs/embedders). For example [SentenceTransformersDocumentEmbedder](https://docs.haystack.deepset.ai/v2.0/docs/sentencetransformersdocumentembedder) can be used in indexing pipeline to calculate document embeddings before writing those to FalkorDB.

## Installation

`falkordb-haystack` can be installed as any other Python library, using pip:

```bash
pip install --upgrade pip # optional
pip install sentence-transformers # required in order to run pipeline examples given below
pip install falkordb-haystack
```

## Usage

### Running FalkorDB

You will need to have a running instance of FalkorDB database to use components from the package.
There are several options available:

- [Docker](https://docs.falkordb.com/getting-started/) - Run FalkorDB in a container
- [FalkorDB Cloud](https://falkordb.com/) - Fully managed cloud instance
- Local installation - See [FalkorDB documentation](https://docs.falkordb.com/)

The simplest way to start database locally will be with Docker container:

```bash
docker run -p 6379:6379 -it --rm falkordb/falkordb:latest
```

FalkorDB 4.0+ supports native vector search for semantic similarity queries. The database uses Redis protocol and runs on port 6379 by default.

> **Note**
> FalkorDB is compatible with Redis protocol. You can use Redis clients to connect to FalkorDB on port 6379. For graph operations and Cypher queries, use the FalkorDB Python client.

### Document Store

Once you have the package installed and the database running, you can start using `FalkorDBDocumentStore` as any other document stores that support embeddings.

```python
from falkordb_haystack import FalkorDBDocumentStore

document_store = FalkorDBDocumentStore(
    host="localhost", port=6379,
    
    
    graph="haystack",
    embedding_dim=384,
    embedding_field="embedding",
    index="document-embeddings", # The name of the Vector Index in FalkorDB
    node_label="Document", # Providing a label to FalkorDB nodes which store Documents
)
```

Alternatively, FalkorDB connection properties could be specified using a dedicated [FalkorDBClientConfig](https://prosto.github.io/falkordb-haystack/reference/falkordb_client/#falkordb_haystack.client.falkordb_client.FalkorDBClientConfig) class:

```python
from falkordb_haystack import FalkorDBClientConfig, FalkorDBDocumentStore

client_config = FalkorDBClientConfig(
    host="localhost", port=6379,
    
    
    graph="haystack",
)

document_store = FalkorDBDocumentStore(client_config=client_config, embedding_dim=384)
```

Assuming there is a list of documents available and a running FalkorDB database you can write/index those in FalkorDB, e.g.:

```python
from haystack import Document

documents = [Document(content="My name is Morgan and I live in Paris.")]

document_store.write_documents(documents)
```

If you intend to obtain embeddings before writing documents use the following code:

```python
from haystack import Document

# import one of the available document embedders
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

documents = [Document(content="My name is Morgan and I live in Paris.")]

document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
document_embedder.warm_up() # will download the model during first run
documents_with_embeddings = document_embedder.run(documents)

document_store.write_documents(documents_with_embeddings.get("documents"))
```

Make sure embedding model produces vectors of same size as it has been set on `FalkorDBDocumentStore`, e.g. setting `embedding_dim=384` would comply with the "sentence-transformers/all-MiniLM-L6-v2" model.

> **Note**
> Most of the time you will be using [Haystack Pipelines](https://docs.haystack.deepset.ai/v2.0/docs/pipelines) to build both indexing and querying RAG scenarios.

It is important to understand how haystack Documents are stored in FalkorDB after you call `write_documents`.

```python
from random import random

sample_embedding = [random() for _ in range(384)]  # using fake/random embedding for brevity here to simplify example
document = Document(
    content="My name is Morgan and I live in Paris.", embedding=sample_embedding, meta={"num_of_years": 3}
)
document.to_dict()
```

The above code converts a Document to a dictionary and will render the following output:

```bash
>>> output:
{
    "id": "11c255ad10bff4286781f596a5afd9ab093ed056d41bca4120c849058e52f24d",
    "content": "My name is Morgan and I live in Paris.",
    "dataframe": None,
    "blob": None,
    "score": None,
    "embedding": [0.025010755222666936, 0.27502931836911926, 0.22321073814882275, ...], # vector of size 384
    "num_of_years": 3,
}
```

The data from the dictionary will be used to create a node in FalkorDB after you write the document with `document_store.write_documents([document])`. You could query it with Cypher, e.g. `MATCH (doc:Document) RETURN doc`. Below is a json representation of the node in FalkorDB:

```js
{
  "identity": 0,
  "labels": [
    "Document" // label name is specified in the FalkorDBDocumentStore.node_label argument
  ],
  "properties": { // this is where Document data is stored
    "id": "11c255ad10bff4286781f596a5afd9ab093ed056d41bca4120c849058e52f24d",
    "embedding": [0.6394268274307251, 0.02501075528562069,0.27502933144569397, ...], // vector of size 384
    "content": "My name is Morgan and I live in Paris.",
    "num_of_years": 3
  },
  "elementId": "4:8bde9fb3-3975-4c3e-9ea1-3e10dbad55eb:0"
}
```

> **Note**
> Metadata (`num_of_years`) is serialized to the same level as rest of attributes (flatten). **It is expected by current implementation** as FalkorDB node's properties can not have nested structures.

The full list of parameters accepted by `FalkorDBDocumentStore` can be found in
[API documentation](https://prosto.github.io/falkordb-haystack/reference/falkordb_store/#falkordb_haystack.document_stores.falkordb_store.FalkorDBDocumentStore.__init__).

### Indexing documents

With Haystack you can use [DocumentWriter](https://docs.haystack.deepset.ai/v2.0/docs/documentwriter) component to write Documents into a Document Store. In the example below we construct pipeline to write documents to FalkorDB using `FalkorDBDocumentStore`:

```python
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.pipeline import Pipeline

from falkordb_haystack import FalkorDBDocumentStore

documents = [Document(content="This is document 1"), Document(content="This is document 2")]

document_store = FalkorDBDocumentStore(
    host="localhost", port=6379,
    
    
    graph="haystack",
    embedding_dim=384,
    embedding_field="embedding",
    index="document-embeddings",
    node_label="Document",
)
embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
document_writer = DocumentWriter(document_store=document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=embedder, name="embedder")
indexing_pipeline.add_component(instance=document_writer, name="writer")

indexing_pipeline.connect("embedder", "writer")
indexing_pipeline.run({"embedder": {"documents": documents}})
```

```bash
>>> output:
`{'writer': {'documents_written': 2}}`
```

### Retrieving documents

`FalkorDBEmbeddingRetriever` component can be used to retrieve documents from FalkorDB by querying vector index using an embedded query. Below is a pipeline which finds documents using query embedding as well as [metadata filtering](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering):

```python
from typing import List

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder

from falkordb_haystack import FalkorDBDocumentStore, FalkorDBEmbeddingRetriever

document_store = FalkorDBDocumentStore(
    host="localhost", port=6379,
    
    
    graph="haystack",
    embedding_dim=384,
    index="document-embeddings",
)

documents = [
    Document(content="My name is Morgan and I live in Paris.", meta={"num_of_years": 3}),
    Document(content="I am Susan and I live in Berlin.", meta={"num_of_years": 7}),
]

# Same model is used for both query and Document embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"

document_embedder = SentenceTransformersDocumentEmbedder(model=model_name)
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)

document_store.write_documents(documents_with_embeddings.get("documents"))

print("Number of documents written: ", document_store.count_documents())

pipeline = Pipeline()
pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model=model_name))
pipeline.add_component("retriever", FalkorDBEmbeddingRetriever(document_store=document_store))
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

result = pipeline.run(
    data={
        "text_embedder": {"text": "What cities do people live in?"},
        "retriever": {
            "top_k": 5,
            "filters": {"field": "num_of_years", "operator": "==", "value": 3},
        },
    }
)

documents: List[Document] = result["retriever"]["documents"]
```

```bash
>>> output:
[Document(id=3930326edabe6d172031557556999e2f8ba258ccde3c876f5e3ac7e66ed3d53a, content: 'My name is Morgan and I live in Paris.', meta: {'num_of_years': 3}, score: 0.8348373770713806)]
```

> **Note**
> You can learn more about how a given metadata filter is converted into Cypher queries by looking at documentation of the [FalkorDBQueryConverter](https://prosto.github.io/falkordb-haystack/reference/metadata_filter/falkordb_query_converter/#falkordb_haystack.metadata_filter.falkordb_query_converter.FalkorDBQueryConverter) class.

### Retrieving documents using Cypher

In certain scenarios you might have an existing graph in FalkorDB database which was created by custom scripts or data ingestion pipelines. The schema of the graph could be complex and not exactly fitting into Haystack Document model. Moreover in many situations you might want to leverage existing graph data to extract more context for grounding LLMs. To make it possible with Haystack we have `FalkorDBDynamicDocumentRetriever` component - a flexible retriever which can run arbitrary Cypher query to obtain documents. This component does not require Document Store to operate.

> **Note**
> The logic of `FalkorDBDynamicDocumentRetriever` could be easily achieved with `FalkorDBQueryReader` + `OutputAdapter` components.
> `FalkorDBDynamicDocumentRetriever` makes sense when you specifically expect Documents as an output of a query execution and would like to avoid additional output conversions in your pipeline (e.g. "FalkorDB Record" --> Document).

The above example of `FalkorDBEmbeddingRetriever` could be rewritten without usage of `FalkorDBDocumentStore` in the retrieval pipeline:

```python
from typing import List

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder

from falkordb_haystack import FalkorDBClientConfig, FalkorDBDocumentStore, FalkorDBDynamicDocumentRetriever

client_config = FalkorDBClientConfig(
    host="localhost", port=6379,
    
    
    graph="haystack",
)

documents = [
    Document(content="My name is Morgan and I live in Paris.", meta={"num_of_years": 3}),
    Document(content="I am Susan and I live in Berlin.", meta={"num_of_years": 7}),
]

# Same model is used for both query and Document embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"

document_embedder = SentenceTransformersDocumentEmbedder(model=model_name)
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)

document_store = FalkorDBDocumentStore(client_config=client_config, embedding_dim=384)
document_store.write_documents(documents_with_embeddings.get("documents"))

# Same model is used for both query and Document embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"

cypher_query = """
            CALL db.index.vector.queryNodes($index, $top_k, $query_embedding)
            YIELD node as doc, score
            MATCH (doc) WHERE doc.num_of_years = $num_of_years
            RETURN doc{.*, score}, score
            ORDER BY score DESC LIMIT $top_k
        """

embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
retriever = FalkorDBDynamicDocumentRetriever(
    client_config=client_config, runtime_parameters=["query_embedding"], doc_node_name="doc"
)

pipeline = Pipeline()
pipeline.add_component("text_embedder", embedder)
pipeline.add_component("retriever", retriever)
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

result = pipeline.run(
    data={
        "text_embedder": {"text": "What cities do people live in?"},
        "retriever": {
            "query": cypher_query,
            "parameters": {"index": "document-embeddings", "top_k": 5, "num_of_years": 3},
        },
    }
)

documents: List[Document] = result["retriever"]["documents"]
```

```bash
>>> output:
[Document(id=4014455c3be5d88151ba12d734a16754d7af75c691dfc3a5f364f81772471bd2, content: 'My name is Morgan and I live in Paris.', meta: {'num_of_years': 3}, score: 0.6696747541427612, embedding: vector of size 384)]
```

Please notice how query parameters are being used in the `cypher_query`:

- `runtime_parameters` is a list of parameter names which are going to be input slots when connecting components
  in a pipeline. In our case `query_embedding` input is connected to the `text_embedder.embedding` output.
- `pipeline.run` specifies additional parameters to the `retriever` component which can be referenced in the
  `cypher_query`, e.g. `top_k` and `num_of_years`.

In some way `FalkorDBDynamicDocumentRetriever` resembles the [PromptBuilder](https://docs.haystack.deepset.ai/v2.0/docs/promptbuilder) component, only instead of prompt it constructs a Cypher query using [parameters](https://falkordb.com/docs/python-manual/current/query-simple/#query-parameters). In the example above documents retrieved by running the query, the `RETURN doc{.*, score}` part returns back found documents with scores. Which node variable is going to be used to construct haystack Document is specified in the `doc_node_name` parameter (see above `doc_node_name="doc"`).

You have options to enhance your RAG pipeline with data having various schemas, for example by first finding nodes using vector search and then expanding query to search for nearby nodes using appropriate Cypher syntax. It is possible to implement "Parent-Child" chunking strategy with such approach. Before that you have to ingest/index data into FalkorDB accordingly by building an indexing pipeline or a custom ingestion script. A simple schema is shown below:

```text
┌────────────┐                ┌─────────────┐
│   Child    │                │   Parent    │
│            │  :HAS_PARENT   │             │
│   content  ├────────────────┤   content   │
│  embedding │                │             │
└────────────┘                └─────────────┘
```

The following Cypher query is an example of how `FalkorDBDynamicDocumentRetriever` can first search embeddings for `Child` document chunks and then **return** `Parent` documents which have larger context window (text length) for RAG applications:

```cypher
// Query Child documents by $query_embedding
CALL db.index.vector.queryNodes($index, $top_k, $query_embedding)
YIELD node as child_doc, score

// Find Parent document for previously retrieved child (e.g. extend RAG context)
MATCH (child_doc)-[:HAS_PARENT]->(parent:Parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent{.*, score}
```

As you might have guessed, the value for the `doc_node_name` parameter should be equal to `parent` according to the query above.

### More examples

You can find more examples in the implementation [repository](https://github.com/prosto/falkordb-haystack/tree/main/examples):

- [indexing_pipeline.py](https://github.com/prosto/falkordb-haystack/blob/main/examples/indexing_pipeline.py) - Indexing text files (documents) from a remote http location.
- [rag_pipeline.py](https://github.com/prosto/falkordb-haystack/blob/main/examples/rag_pipeline.py) - Generative question answering RAG pipeline using `FalkorDBEmbeddingRetriever` to fetch documents from FalkorDB document store and answer question using [HuggingFaceTGIGenerator](https://docs.haystack.deepset.ai/v2.0/docs/huggingfacetgigenerator).
- [rag_pipeline_cypher.py](https://github.com/prosto/falkordb-haystack/blob/main/examples/rag_pipeline_cypher.py) - Same as `rag_pipeline.py` but using `FalkorDBDynamicDocumentRetriever`.

More technical details available in the [Code Reference](https://prosto.github.io/falkordb-haystack/reference/falkordb_store/) documentation. For example, in real world scenarios there could be requirements to tune connection settings to FalkorDB database (e.g. request timeout). [FalkorDBDocumentStore](https://prosto.github.io/falkordb-haystack/reference/falkordb_store/#falkordb_haystack.document_stores.FalkorDBDocumentStore.__init__) accepts an extended client configuration using [FalkorDBClientConfig](https://prosto.github.io/falkordb-haystack/reference/falkordb_client/#falkordb_haystack.client.falkordb_client.FalkorDBClientConfig) class.

## License

`falkordb-haystack` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
