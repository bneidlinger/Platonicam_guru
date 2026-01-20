"""
PDF Ingestion Page - Upload and process vendor documentation.
"""
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import Settings
from src.ingest import IngestionPipeline
from src.vectorstore.chroma_store import ChromaStore


st.set_page_config(
    page_title="Document Ingestion",
    page_icon="📁",
    layout="wide",
)

st.title("📁 Document Ingestion")
st.write("Upload and process vendor PDFs to add them to the knowledge base.")


# =============================================================================
# CACHED RESOURCES
# =============================================================================

@st.cache_resource
def get_store():
    return ChromaStore()


@st.cache_resource
def get_pipeline():
    return IngestionPipeline()


# =============================================================================
# DATABASE STATUS
# =============================================================================

def render_status():
    """Show current database status."""
    store = get_store()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Chunks", store.count())

    stats = store.get_stats()

    with col2:
        vendors = stats.get("by_vendor", {})
        vendor_str = ", ".join(f"{k}: {v}" for k, v in vendors.items()) or "None"
        st.metric("By Vendor", len(vendors))
        st.caption(vendor_str)

    with col3:
        doc_types = stats.get("by_doc_type", {})
        st.metric("Doc Types", len(doc_types))


render_status()

st.divider()


# =============================================================================
# PDF UPLOAD
# =============================================================================

st.subheader("📤 Upload PDFs")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload vendor datasheets, installation guides, or accessory documentation.",
    )

with col2:
    vendor = st.selectbox(
        "Vendor",
        Settings.VENDORS + ["unknown"],
        help="Select the vendor for these documents.",
    )

if uploaded_files:
    st.write(f"**{len(uploaded_files)} file(s) selected:**")
    for f in uploaded_files:
        st.caption(f"• {f.name} ({f.size / 1024:.1f} KB)")

    if st.button("🚀 Process Files", type="primary", use_container_width=True):
        pipeline = get_pipeline()

        # Check prerequisites
        if not pipeline.embedder.check_model_available():
            st.error(f"Embedding model not available. Run: `ollama pull {pipeline.embedder.model}`")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_chunks = 0
            total_images = 0

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}")

                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp.name)

                try:
                    # Process the PDF
                    file_stats = pipeline.ingest_pdf(
                        tmp_path,
                        vendor=vendor,
                        skip_existing=True,
                    )

                    total_chunks += file_stats["added"]
                    total_images += file_stats["images"]

                    st.success(f"✓ {uploaded_file.name}: {file_stats['added']} chunks, {file_stats['images']} images")

                except Exception as e:
                    st.error(f"✗ {uploaded_file.name}: {str(e)}")

                finally:
                    # Cleanup temp file
                    tmp_path.unlink(missing_ok=True)

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("Complete!")
            st.balloons()

            # Show summary
            st.info(f"Added {total_chunks} chunks and extracted {total_images} images.")

            # Clear cache to refresh stats
            st.cache_resource.clear()
            st.rerun()


st.divider()


# =============================================================================
# BATCH INGESTION
# =============================================================================

st.subheader("📂 Batch Ingestion from Folder")
st.write("Process all PDFs in the data directory.")

col1, col2, col3 = st.columns(3)

with col1:
    batch_vendor = st.selectbox(
        "Vendor Filter",
        ["all"] + Settings.VENDORS,
        key="batch_vendor",
    )

with col2:
    force_reprocess = st.checkbox("Force reprocess (ignore existing)")

with col3:
    clear_first = st.checkbox("Clear database first", help="⚠️ This will delete all existing data!")

# Show folder contents
data_dir = Settings.DATA_DIR
st.caption(f"Data directory: `{data_dir}`")

pdf_count = 0
for vendor_name in Settings.VENDORS:
    vendor_dir = data_dir / vendor_name
    if vendor_dir.exists():
        pdfs = list(vendor_dir.glob("*.pdf"))
        if pdfs:
            st.caption(f"  • {vendor_name}/: {len(pdfs)} PDFs")
            pdf_count += len(pdfs)

if pdf_count == 0:
    st.warning("No PDFs found in data directory. Add files to `data/pdfs/<vendor>/`")
else:
    if st.button("🚀 Run Batch Ingestion", use_container_width=True):
        pipeline = get_pipeline()

        if not pipeline.embedder.check_model_available():
            st.error(f"Embedding model not available. Run: `ollama pull {pipeline.embedder.model}`")
        else:
            if clear_first:
                with st.spinner("Clearing database..."):
                    pipeline.store.clear()
                    st.info("Database cleared.")

            with st.spinner("Processing PDFs..."):
                vendor_filter = None if batch_vendor == "all" else batch_vendor

                stats = pipeline.ingest_directory(
                    vendor=vendor_filter,
                    skip_existing=not force_reprocess,
                )

            st.success(f"""
            **Ingestion Complete!**
            - PDFs processed: {stats['pdfs_processed']}
            - Chunks stored: {stats['chunks_stored']}
            - Images extracted: {stats['images_extracted']}
            """)

            st.cache_resource.clear()
            st.rerun()


st.divider()


# =============================================================================
# DATABASE MANAGEMENT
# =============================================================================

st.subheader("🗄️ Database Management")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Refresh Stats", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

with col2:
    if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
        if st.session_state.get("confirm_clear"):
            store = get_store()
            count = store.clear()
            st.success(f"Cleared {count} chunks.")
            st.session_state.confirm_clear = False
            st.cache_resource.clear()
            st.rerun()
        else:
            st.session_state.confirm_clear = True
            st.warning("Click again to confirm clearing all data.")
