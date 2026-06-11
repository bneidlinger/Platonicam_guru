"""Diagnostic: exercise the RAG pipeline routes end-to-end (retrieval layer focus)."""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag.prompts import classify_query
from src.rag.retriever import Retriever
from src.parser.metadata_extractor import MetadataExtractor

ext = MetadataExtractor()
retriever = Retriever()

print("=== 1. Query classification ===")
queries = [
    "What is the power consumption of the M1075-L?",
    "What mount fits the M1135-E?",
    "Compare the M1075-L and M1055-L",
    "What resolution is the Q1808-LE?",
    "Tell me about the M1075-L camera",
    "Can the Q6075 do auto tracking?",
]
for q in queries:
    print(f"  [{classify_query(q):>13}] {q}")

print("\n=== 2. Model extraction from user-style queries ===")
for q in [
    "What is the power consumption of the M1075-L?",
    "tell me about the m1075 camera",          # partial, lowercase
    "specs for the M1075-L MK II please",      # MK II variant
    "does it support wdr and ir illumination at 30 fps",  # no model
    "I need a camera for a parking lot about 100ft away", # no model
]:
    print(f"  {ext.extract_model_numbers(q)!r:30} <- {q!r}")

print("\n=== 3. General-route retrieval: exact model (M1075-L) ===")
results = retriever.retrieve_for_models(["M1075-L"], "power consumption")
print(f"  retrieve_for_models(['M1075-L']) -> {len(results)} chunks")
for r in results[:3]:
    m = r["metadata"]
    print(f"    {m.get('source_file')} p{m.get('page_num')} model={m.get('model_num')} poe={m.get('poe_wattage')} dist={r.get('distance'):.3f}")

print("\n=== 4. General-route retrieval: partial model (M1075) ===")
results = retriever.retrieve_for_models(["M1075"], "tell me about this camera")
print(f"  retrieve_for_models(['M1075']) -> {len(results)} chunks  <-- empty means dead-end, no fallback")

print("\n=== 5. Accessory route WITH a model (multi-key where filter) ===")
try:
    ctx = retriever.retrieve_accessory_context("What mount fits?", model_num="M1135-E")
    print(f"  OK: {len(ctx['results'])} results")
except Exception as e:
    print(f"  CRASH: {type(e).__name__}: {e}")
    traceback.print_exc(limit=2)

print("\n=== 6. POE route: verified data injected into prompt ===")
ctx = retriever.retrieve_poe_context("What is the power consumption of the M1075-L?")
print(f"  models extracted: {ctx['models']}")
print(f"  poe_data block:\n{ctx['poe_data']}")

print("\n=== 7. Pure semantic retrieval quality (no model in query) ===")
for q in [
    "outdoor PTZ camera with 30x zoom for a stadium",
    "explosion protected camera for oil and gas site",
    "what is the widest field of view fisheye option",
]:
    results = retriever.retrieve(q)
    print(f"  Q: {q}")
    for r in results[:3]:
        m = r["metadata"]
        print(f"    dist={r.get('distance'):.3f}  {m.get('source_file')} p{m.get('page_num')}")

print("\n=== 8. Wattage regex spot checks (doc-style strings) ===")
for s in [
    "Power consumption: 12.95 W PoE",        # real, followed by PoE
    "Max 25.5W power consumption",           # real, followed by 'power'
    "Typical 6.5 W, max 12.9 W",             # real, plain
    "1080p with WDR",                        # junk: 'with'
    "delivers 95 white LEDs",                # junk: 'white'
    "2 wire connection",                     # junk: 'wire'
]:
    print(f"  {ext.extract_poe_wattage(s)!r:8} <- {s!r}")
