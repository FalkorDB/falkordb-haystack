from typing import Optional

from haystack.document_stores.errors import DocumentStoreError


class FalkorDBDocumentStoreError(DocumentStoreError):
    """Error for issues that occur in a FalkorDB Document Store"""

    def __init__(self, message: Optional[str] = None):
        super().__init__(message=message)


class FalkorDBClientError(DocumentStoreError):
    """Error for issues that occur in a FalkorDB client"""

    def __init__(self, message: Optional[str] = None):
        super().__init__(message)


class FalkorDBFilterParserError(Exception):
    """Error is raised when metadata filters are failing to parse"""

    pass
