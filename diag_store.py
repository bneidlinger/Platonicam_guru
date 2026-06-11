"""Diagnostic: inspect ChromaDB contents and metadata quality."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.vectorstore.chroma_store import ChromaStore

store = ChromaStore()
print(f"Total chunks: {store.count()}")

data = store.collection.get(include=["metadatas"])
metas = data.get("metadatas", [])

vendors = Counter(m.get("vendor", "none") for m in metas)
doc_types = Counter(m.get("doc_type", "none") for m in metas)
print(f"\nBy vendor: {dict(vendors)}")
print(f"By doc_type: {dict(doc_types)}")

with_model = [m for m in metas if m.get("model_num")]
with_poe = [m for m in metas if m.get("poe_wattage") is not None]
print(f"\nChunks with model_num: {len(with_model)}/{len(metas)} ({100*len(with_model)/max(len(metas),1):.0f}%)")
print(f"Chunks with poe_wattage: {len(with_poe)}/{len(metas)} ({100*len(with_poe)/max(len(metas),1):.0f}%)")

model_counts = Counter(m["model_num"] for m in with_model)
print(f"\nDistinct model_num values: {len(model_counts)}")
print("Top 40 model_num values:")
for model, count in model_counts.most_common(40):
    print(f"  {model}: {count}")

# Source files vs models: how many source files have a chunk tagged with the filename model?
sources = Counter(m.get("source_file", "?") for m in metas)
print(f"\nDistinct source files in store: {len(sources)}")

# POE wattage sanity
poe_vals = sorted(float(m["poe_wattage"]) for m in with_poe)
if poe_vals:
    print(f"\nPOE wattage range: {poe_vals[0]} - {poe_vals[-1]}")
    print(f"POE sample values: {poe_vals[:10]} ... {poe_vals[-10:]}")

# Spot-check: known models from filenames
print("\n--- Exact metadata filter checks ---")
for model in ["M1075-L", "M1135-E", "Q6075", "VB-H47", "P3807", "F4105-LRE", "M1055-L"]:
    docs = store.get_all_by_model(model)
    poe = store.get_poe_wattage(model)
    print(f"  model_num={model!r}: {len(docs)} chunks, poe_wattage={poe}")
