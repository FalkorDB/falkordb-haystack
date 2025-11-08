# Migration Guide: Neo4j to FalkorDB

This document provides a comprehensive guide for migrating from the previous Neo4j integration to the new FalkorDB integration.

## Overview

FalkorDB is a high-performance graph database built for AI applications, featuring ultra-fast query execution and native vector search support. This integration replaces the Neo4j backend while maintaining compatibility with the Haystack framework and OpenCypher query language.

## Key Differences

### Package and Naming

| Component | Neo4j | FalkorDB |
|-----------|-------|----------|
| Package Name | `neo4j-haystack` | `falkordb-haystack` |
| Module Name | `neo4j_haystack` | `falkordb_haystack` |
| Document Store | `Neo4jDocumentStore` | `FalkorDBDocumentStore` |
| Client | `Neo4jClient` | `FalkorDBClient` |
| Retriever | `Neo4jEmbeddingRetriever` | `FalkorDBEmbeddingRetriever` |

### Connection Parameters

**Neo4j:**
```python
from neo4j_haystack import Neo4jDocumentStore

store = Neo4jDocumentStore(
    url="bolt://localhost:7687",
    database="neo4j",
    username="neo4j",
    password="password"
)
```

**FalkorDB:**
```python
from falkordb_haystack import FalkorDBDocumentStore

store = FalkorDBDocumentStore(
    host="localhost",
    port=6379,
    graph="haystack",
    username=None,  # Optional
    password=None   # Optional
)
```

### Database Setup

**Neo4j:**
```bash
docker run -p 7474:7474 -p 7687:7687 \
    --env NEO4J_AUTH=neo4j/password \
    neo4j:5.15.0
```

**FalkorDB:**
```bash
docker run -p 6379:6379 \
    falkordb/falkordb:latest
```

### Protocol and Port

| Database | Protocol | Default Port | URL Format |
|----------|----------|--------------|------------|
| Neo4j | Bolt | 7687 | `bolt://host:port` |
| FalkorDB | Redis | 6379 | host/port (no URL) |

## Step-by-Step Migration

### 1. Update Dependencies

**pyproject.toml / requirements.txt:**
```diff
- neo4j-haystack>=2.2.0
+ falkordb-haystack>=1.0.0
```

### 2. Update Imports

```python
# Before
from neo4j_haystack import (
    Neo4jDocumentStore,
    Neo4jEmbeddingRetriever,
    Neo4jClientConfig,
)

# After
from falkordb_haystack import (
    FalkorDBDocumentStore,
    FalkorDBEmbeddingRetriever,
    FalkorDBClientConfig,
)
```

### 3. Update Configuration

**Using Direct Parameters:**

```python
# Before
document_store = Neo4jDocumentStore(
    url="bolt://localhost:7687",
    database="neo4j",
    username="neo4j",
    password="passw0rd",
    embedding_dim=384,
    index="document-embeddings",
)

# After
document_store = FalkorDBDocumentStore(
    host="localhost",
    port=6379,
    graph="haystack",
    embedding_dim=384,
    index="document-embeddings",
)
```

**Using Client Config:**

```python
# Before
from neo4j_haystack import Neo4jClientConfig

config = Neo4jClientConfig(
    url="bolt://localhost:7687",
    database="neo4j",
    username="neo4j",
    password="passw0rd",
)

# After
from falkordb_haystack import FalkorDBClientConfig

config = FalkorDBClientConfig(
    host="localhost",
    port=6379,
    graph="haystack",
)
```

### 4. Update Environment Variables

```bash
# Before
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_DATABASE="neo4j"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"

# After
export FALKORDB_HOST="localhost"
export FALKORDB_PORT="6379"
export FALKORDB_GRAPH="haystack"
```

### 5. Update Pipeline Code

**Indexing Pipeline:**

```python
# Before
from neo4j_haystack import Neo4jDocumentStore
from haystack.components.writers import DocumentWriter

document_store = Neo4jDocumentStore(
    url="bolt://localhost:7687",
    database="neo4j",
    embedding_dim=384,
)

# After
from falkordb_haystack import FalkorDBDocumentStore
from haystack.components.writers import DocumentWriter

document_store = FalkorDBDocumentStore(
    host="localhost",
    port=6379,
    graph="haystack",
    embedding_dim=384,
)
```

**Retrieval Pipeline:**

```python
# Before
from neo4j_haystack import Neo4jEmbeddingRetriever

retriever = Neo4jEmbeddingRetriever(document_store=document_store)

# After
from falkordb_haystack import FalkorDBEmbeddingRetriever

retriever = FalkorDBEmbeddingRetriever(document_store=document_store)
```

## API Compatibility

### What Stays the Same

- ✅ **Cypher Queries**: All OpenCypher queries remain compatible
- ✅ **Document Model**: Haystack Document structure unchanged
- ✅ **Embeddings**: Vector embeddings work the same way
- ✅ **Metadata Filtering**: Filter syntax unchanged
- ✅ **Component Interfaces**: All Haystack component protocols maintained

### What Changes

- ❌ **Connection Protocol**: Bolt → Redis
- ❌ **Connection Parameters**: URL-based → host/port-based
- ❌ **Database Selection**: `database` parameter → `graph` parameter
- ❌ **Authentication**: Neo4j auth → Optional (FalkorDB uses Redis protocol)

## Performance Considerations

FalkorDB is optimized for AI workloads and typically offers:

- **Faster Query Execution**: 10x-300x faster for certain query types
- **Lower Latency**: Sub-140ms p99 response times
- **Better Multi-tenancy**: Native support for multiple graphs
- **Vector Search**: Native vector indexing from version 4.0+

## Troubleshooting

### Common Issues

**1. Connection Refused**
```python
# Make sure FalkorDB is running on the correct port
docker ps | grep falkordb

# Check connection parameters
config = FalkorDBClientConfig(
    host="localhost",  # Not "bolt://localhost"
    port=6379,         # Default FalkorDB port
    graph="haystack",  # Graph name
)
```

**2. Import Errors**
```python
# Make sure you've installed the correct package
pip uninstall neo4j-haystack
pip install falkordb-haystack
```

**3. Vector Index Issues**
```python
# FalkorDB uses different syntax for vector indexes
# The package handles this automatically, but if you're
# using custom queries, update to FalkorDB syntax:

# Neo4j:
# CALL db.index.vector.createNodeIndex(...)

# FalkorDB:
# CREATE VECTOR INDEX FOR (n:Label) ON (n.property)
# OPTIONS {dimension: 384, similarityFunction: 'cosine'}
```

## Testing Your Migration

### Basic Connectivity Test

```python
from falkordb_haystack import FalkorDBClient, FalkorDBClientConfig

config = FalkorDBClientConfig(
    host="localhost",
    port=6379,
    graph="test"
)

client = FalkorDBClient(config)
try:
    client.verify_connectivity()
    print("✅ Connection successful!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
finally:
    client.close_driver()
```

### Document Store Test

```python
from falkordb_haystack import FalkorDBDocumentStore
from haystack import Document

# Create document store
document_store = FalkorDBDocumentStore(
    host="localhost",
    port=6379,
    graph="test",
    embedding_dim=384,
)

# Test write
documents = [
    Document(content="Test document", embedding=[0.1] * 384)
]
document_store.write_documents(documents)

# Test count
count = document_store.count_documents()
print(f"✅ Documents stored: {count}")

# Test retrieval
docs = document_store.filter_documents()
print(f"✅ Documents retrieved: {len(docs)}")
```

## Support and Resources

- **FalkorDB Documentation**: https://docs.falkordb.com/
- **FalkorDB GitHub**: https://github.com/FalkorDB/FalkorDB
- **Haystack Documentation**: https://docs.haystack.deepset.ai/
- **Package Repository**: https://github.com/FalkorDB/falkordb-haystack
- **Issues**: https://github.com/FalkorDB/falkordb-haystack/issues

## Additional Notes

### Graph vs Database

In Neo4j, you connect to a "database". In FalkorDB, you connect to a "graph" within the database. This is similar to how you select a Redis database, but for graph data.

### Redis Protocol

FalkorDB uses the Redis protocol, which means:
- No Bolt driver required
- Simpler connection setup
- Compatible with Redis tools and clients
- Port 6379 by default (standard Redis port)

### Vector Search

Both databases support vector search, but with different implementations:
- **Neo4j**: Uses `db.index.vector.*` procedures
- **FalkorDB**: Uses native `CREATE VECTOR INDEX` and `db.idx.vector.*` procedures

The package abstracts these differences, so your code remains the same.
