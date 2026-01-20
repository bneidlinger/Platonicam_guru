"""
Streamlit Web UI for the Surveillance Design Assistant.

Features:
- Chat interface with RAG-powered responses
- Vendor filtering
- Project mode with camera tracking
- POE budget calculator
- Export to CSV/JSON
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from src.rag.chain import RAGChain
from src.rag.memory import ConversationMemory
from src.vectorstore.chroma_store import ChromaStore


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Surveillance Design Assistant",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = RAGChain()

    if "vendor_filter" not in st.session_state:
        st.session_state.vendor_filter = "all"

    if "project_cameras" not in st.session_state:
        st.session_state.project_cameras = []

    if "project_name" not in st.session_state:
        st.session_state.project_name = "New Project"

    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True


init_session_state()


# =============================================================================
# CACHED RESOURCES
# =============================================================================

@st.cache_resource
def get_store():
    """Get ChromaDB store (cached)."""
    return ChromaStore()


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar with filters and project controls."""
    with st.sidebar:
        st.title("📹 Design Assistant")

        # Store status
        store = get_store()
        chunk_count = store.count()

        if chunk_count == 0:
            st.warning("⚠️ No documents loaded. Run ingestion first.")
        else:
            st.success(f"✓ {chunk_count} chunks indexed")

        st.divider()

        # Vendor Filter
        st.subheader("🏭 Vendor Filter")
        vendor_options = ["all"] + Settings.VENDORS
        vendor = st.selectbox(
            "Filter by vendor",
            vendor_options,
            index=vendor_options.index(st.session_state.vendor_filter),
            key="vendor_select",
        )

        if vendor != st.session_state.vendor_filter:
            st.session_state.vendor_filter = vendor
            if vendor != "all":
                st.session_state.rag_chain.set_vendor_filter(vendor)
            else:
                st.session_state.rag_chain.clear_vendor_filter()

        st.divider()

        # Project Mode
        render_project_panel()

        st.divider()

        # Settings
        st.subheader("⚙️ Settings")
        st.session_state.show_sources = st.checkbox(
            "Show source citations",
            value=st.session_state.show_sources,
        )

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.rag_chain.clear_memory()
            st.rerun()


def render_project_panel():
    """Render the project tracking panel."""
    st.subheader("📋 Project Mode")

    # Project name
    st.session_state.project_name = st.text_input(
        "Project Name",
        value=st.session_state.project_name,
    )

    # Add camera
    with st.expander("➕ Add Camera", expanded=False):
        new_camera = st.text_input(
            "Model Number",
            placeholder="e.g., XNV-8080R",
            key="new_camera_input",
        )
        quantity = st.number_input("Quantity", min_value=1, value=1, key="quantity_input")

        if st.button("Add to Project", use_container_width=True):
            if new_camera:
                add_camera_to_project(new_camera.upper(), quantity)
                st.rerun()

    # Camera list
    if st.session_state.project_cameras:
        st.write("**Cameras:**")
        for i, cam in enumerate(st.session_state.project_cameras):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"{cam['model']}")
            with col2:
                st.write(f"x{cam['quantity']}")
            with col3:
                if st.button("✕", key=f"remove_{i}"):
                    st.session_state.project_cameras.pop(i)
                    st.rerun()

        # POE Budget
        st.divider()
        render_poe_budget()

        # Export buttons
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Export CSV", use_container_width=True):
                export_project_csv()
        with col2:
            if st.button("📋 Export JSON", use_container_width=True):
                export_project_json()
    else:
        st.caption("No cameras added yet.")


def add_camera_to_project(model: str, quantity: int):
    """Add a camera to the project list."""
    # Check if already exists
    for cam in st.session_state.project_cameras:
        if cam["model"] == model:
            cam["quantity"] += quantity
            return

    # Get POE info from store
    store = get_store()
    poe_wattage = store.get_poe_wattage(model)

    st.session_state.project_cameras.append({
        "model": model,
        "quantity": quantity,
        "poe_wattage": poe_wattage,
    })


def render_poe_budget():
    """Render POE budget calculation."""
    st.write("**⚡ POE Budget**")

    total_watts = 0
    missing_data = []

    for cam in st.session_state.project_cameras:
        if cam["poe_wattage"]:
            watts = cam["poe_wattage"] * cam["quantity"]
            total_watts += watts
            st.caption(f"{cam['model']}: {cam['poe_wattage']}W × {cam['quantity']} = {watts:.1f}W")
        else:
            missing_data.append(cam["model"])

    st.metric("Total Power", f"{total_watts:.1f}W")

    if missing_data:
        st.warning(f"Missing data: {', '.join(missing_data)}")


def export_project_csv():
    """Export project as CSV."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Model", "Quantity", "POE Wattage (W)", "Total Watts"])

    for cam in st.session_state.project_cameras:
        poe = cam["poe_wattage"] or "N/A"
        total = (cam["poe_wattage"] * cam["quantity"]) if cam["poe_wattage"] else "N/A"
        writer.writerow([cam["model"], cam["quantity"], poe, total])

    csv_data = output.getvalue()

    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"{st.session_state.project_name.replace(' ', '_')}_BOM.csv",
        mime="text/csv",
    )


def export_project_json():
    """Export project as JSON."""
    total_watts = sum(
        (cam["poe_wattage"] or 0) * cam["quantity"]
        for cam in st.session_state.project_cameras
    )

    export_data = {
        "project_name": st.session_state.project_name,
        "export_date": datetime.now().isoformat(),
        "cameras": st.session_state.project_cameras,
        "total_poe_watts": total_watts,
    }

    json_data = json.dumps(export_data, indent=2)

    st.download_button(
        label="Download JSON",
        data=json_data,
        file_name=f"{st.session_state.project_name.replace(' ', '_')}_BOM.json",
        mime="application/json",
    )


# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================

def render_chat():
    """Render the main chat interface."""
    st.title("Surveillance Design Assistant")

    # Vendor badge
    if st.session_state.vendor_filter != "all":
        st.caption(f"🏭 Filtering: {st.session_state.vendor_filter.upper()}")

    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if enabled
            if message["role"] == "assistant" and st.session_state.show_sources:
                if message.get("sources"):
                    with st.expander("📚 Sources", expanded=False):
                        for source in message["sources"]:
                            st.caption(f"• {source}")

    # Chat input
    if prompt := st.chat_input("Ask about cameras, specs, accessories, or POE..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                try:
                    # Get vendor filter
                    vendor = None
                    if st.session_state.vendor_filter != "all":
                        vendor = st.session_state.vendor_filter

                    # Query RAG chain
                    response = st.session_state.rag_chain.query(
                        prompt,
                        vendor=vendor,
                        stream=False,
                    )

                    st.markdown(response)

                    # Get sources from memory
                    sources = []
                    if st.session_state.rag_chain.memory.messages:
                        last_msg = st.session_state.rag_chain.memory.messages[-1]
                        sources = last_msg.metadata.get("sources", [])

                    # Store message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


# =============================================================================
# QUICK ACTIONS
# =============================================================================

def render_quick_actions():
    """Render quick action buttons."""
    st.subheader("Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 POE Budget Help", use_container_width=True):
            inject_query("How do I calculate POE budget for a camera system?")

    with col2:
        if st.button("🔧 Mount Options", use_container_width=True):
            inject_query("What mounting options are available for dome cameras?")

    with col3:
        if st.button("📋 Compare Models", use_container_width=True):
            inject_query("How do I compare different camera models?")

    with col4:
        if st.button("⚙️ Config Defaults", use_container_width=True):
            inject_query("What are the default IP settings for cameras?")


def inject_query(query: str):
    """Inject a query into the chat."""
    st.session_state.messages.append({"role": "user", "content": query})
    st.rerun()


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    render_sidebar()

    # Main content
    render_chat()

    # Quick actions at bottom
    st.divider()
    render_quick_actions()


if __name__ == "__main__":
    main()
