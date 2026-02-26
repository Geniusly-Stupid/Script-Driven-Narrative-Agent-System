from app.vector_store import ChromaStore

store = ChromaStore(".chroma")
collection = store.collection

data = collection.get(include=["documents", "metadatas"])
docs = data.get("documents", [])
metas = data.get("metadatas", [])

print(f"Total records: {len(docs)}")
for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
    print(f"\n[{i}]")
    print(f"type: {meta.get('type')}")
    print(f"name: {meta.get('name')}")
    print(f"scene_id: {meta.get('scene_id')}")
    print(f"plot_id: {meta.get('plot_id')}")
    print(f"document: {doc}")
