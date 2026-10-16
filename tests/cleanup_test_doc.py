import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import vector_store

TEST_DOC_NAMES = [
    "integration_test.pdf",
    "multimodal_test.pdf",
    "test_paper.pdf",
    "Paper_A.pdf",
    "Paper_B.pdf",
]

collection = vector_store.get_collection()
before = vector_store.list_documents()
for name in TEST_DOC_NAMES:
    collection.delete(where={"document_name": name})
after = vector_store.list_documents()

print("Before:", before)
print("After:", after)
