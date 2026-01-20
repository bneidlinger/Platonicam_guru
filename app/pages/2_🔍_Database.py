"""
Database Explorer Page - Browse and search the document database.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import Settings
from src.vectorstore.chroma_store import ChromaStore
from src.embeddings.ollama_embed import OllamaEmbedder


st.set_page_config(
    page_title="Database Explorer",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Database Explorer")
st.write("Browse and search the indexed documentation.")


# =============================================================================
# CACHED RESOURCES
# =============================================================================

@st.cache_resource
def get_store():
    return ChromaStore()


@st.cache_resource
def get_embedder():
    return OllamaEmbedder()


# =============================================================================
# STATISTICS
# =============================================================================

store = get_store()
stats = store.get_stats()

st.subheader("📊 Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Chunks", stats["total_chunks"])

with col2:
    st.write("**By Vendor:**")
    for vendor, count in stats.get("by_vendor", {}).items():
        st.caption(f"• {vendor}: {count}")

with col3:
    st.write("**By Document Type:**")
    for doc_type, count in stats.get("by_doc_type", {}).items():
        st.caption(f"• {doc_type}: {count}")

st.divider()


# =============================================================================
# SEARCH
# =============================================================================

st.subheader("🔎 Search Documents")

col1, col2 = st.columns([3, 1])

with col1:
    search_query = st.text_input(
        "Search query",
        placeholder="e.g., XNV-8080R power consumption",
    )

with col2:
    vendor_filter = st.selectbox(
        "Vendor",
        ["all"] + Settings.VENDORS,
        key="search_vendor",
    )

col1, col2 = st.columns([1, 1])

with col1:
    num_results = st.slider("Results", 1, 20, 5)

with col2:
    doc_type_filter = st.selectbox(
        "Document Type",
        ["all", "datasheet", "installation", "accessory", "manual", "guide"],
    )

if st.button("🔍 Search", type="primary") and search_query:
    embedder = get_embedder()

    with st.spinner("Searching..."):
        # Generate query embedding
        query_embedding = embedder.embed_text(search_query)

        # Build filter
        where = {}
        if vendor_filter != "all":
            where["vendor"] = vendor_filter
        if doc_type_filter != "all":
            where["doc_type"] = doc_type_filter

        # Search
        results = store.search(
            query_embedding=query_embedding,
            n_results=num_results,
            where=where if where else None,
        )

    if results:
        st.success(f"Found {len(results)} results")

        for i, result in enumerate(results):
            with st.expander(
                f"**Result {i+1}** - {result['metadata'].get('source_file', 'Unknown')} "
                f"(Distance: {result.get('distance', 0):.4f})",
                expanded=(i == 0),
            ):
                meta = result.get("metadata", {})

                # Metadata badges
                cols = st.columns(5)
                with cols[0]:
                    if meta.get("vendor"):
                        st.caption(f"🏭 {meta['vendor'].upper()}")
                with cols[1]:
                    if meta.get("model_num"):
                        st.caption(f"📷 {meta['model_num']}")
                with cols[2]:
                    if meta.get("page_num"):
                        st.caption(f"📄 Page {meta['page_num']}")
                with cols[3]:
                    if meta.get("poe_wattage"):
                        st.caption(f"⚡ {meta['poe_wattage']}W")
                with cols[4]:
                    if meta.get("doc_type"):
                        st.caption(f"📁 {meta['doc_type']}")

                # Content
                st.markdown("---")
                st.write(result.get("content", ""))

                # Image references
                image_refs = meta.get("image_refs", "")
                if image_refs:
                    st.markdown("---")
                    st.caption("🖼️ Associated images:")
                    if isinstance(image_refs, str):
                        images = image_refs.split(",")
                    else:
                        images = image_refs

                    for img_path in images[:3]:  # Limit to 3
                        img_path = img_path.strip()
                        if Path(img_path).exists():
                            st.image(img_path, width=200)
                        else:
                            st.caption(f"  • {img_path}")
    else:
        st.warning("No results found.")

st.divider()


# =============================================================================
# POE LOOKUP
# =============================================================================

st.subheader("⚡ POE Lookup")
st.write("Look up power consumption for specific models.")

model_input = st.text_input(
    "Model numbers (comma-separated)",
    placeholder="e.g., XNV-8080R, P3265-LVE",
)

if st.button("⚡ Calculate POE") and model_input:
    models = [m.strip().upper() for m in model_input.split(",")]

    budget = store.calculate_poe_budget(models)

    if budget["by_model"]:
        st.write("**Power Consumption:**")

        for model, watts in budget["by_model"].items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {model}")
            with col2:
                st.write(f"{watts}W")

        st.metric("Total", f"{budget['total_watts']:.1f}W")

        if budget["missing"]:
            st.warning(f"Missing data for: {', '.join(budget['missing'])}")
    else:
        st.warning("No POE data found for these models.")

st.divider()


# =============================================================================
# BROWSE BY MODEL
# =============================================================================

st.subheader("📷 Browse by Model")

model_search = st.text_input(
    "Model number",
    placeholder="e.g., XNV-8080R",
    key="model_browse",
)

if st.button("📷 Find Model") and model_search:
    docs = store.get_all_by_model(model_search.upper())

    if docs:
        st.success(f"Found {len(docs)} chunks for {model_search.upper()}")

        for i, doc in enumerate(docs):
            meta = doc.get("metadata", {})
            with st.expander(f"Chunk {i+1} - {meta.get('source_file', 'Unknown')}"):
                st.write(doc.get("content", ""))
    else:
        st.warning(f"No documents found for model: {model_search}")
