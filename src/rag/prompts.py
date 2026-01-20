"""
Prompt templates for the Surveillance Design Assistant.

Design Principle: LLMs for summarization and natural language.
Numerical data (POE wattage, etc.) comes from metadata, not generation.
"""

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are a Surveillance Design Assistant for Physical Security Systems Engineers.

Your role is to help engineers:
- Find technical specifications for IP cameras (Hanwha, Axis, Bosch)
- Identify compatible accessories (mounts, brackets, pendants)
- Answer questions about camera capabilities and configurations
- Assist with system design decisions

Guidelines:
1. Always cite your sources with the document name and page number
2. Be precise with technical specifications - never guess values
3. If information is not in the provided context, say so clearly
4. For numerical values (power, resolution, etc.), only state what's in the documentation
5. When multiple options exist, present them clearly for the engineer to decide

You have access to vendor documentation that will be provided as context."""

SYSTEM_PROMPT_CONCISE = """You are a technical assistant for security system engineers.
Answer questions about IP cameras using only the provided context.
Always cite sources. Never guess specifications."""


# =============================================================================
# RAG CONTEXT TEMPLATE
# =============================================================================

RAG_TEMPLATE = """Use the following documentation excerpts to answer the question.
If the answer is not in the context, say "I don't have that information in the loaded documentation."

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, technical answer with source citations."""

RAG_TEMPLATE_WITH_METADATA = """Use the following documentation excerpts to answer the question.

CONTEXT:
{context}

EXTRACTED DATA (verified from documents):
{metadata_summary}

QUESTION: {question}

Instructions:
- Use the extracted data for any numerical values (power, resolution, etc.)
- Cite sources as [Document Name, Page X]
- If information conflicts, note the discrepancy"""


# =============================================================================
# SPECIALIZED PROMPTS
# =============================================================================

POE_QUERY_TEMPLATE = """The engineer needs power consumption information.

CONTEXT FROM DOCUMENTATION:
{context}

VERIFIED POE DATA (from document metadata):
{poe_data}

QUESTION: {question}

Instructions:
- Use the VERIFIED POE DATA for exact wattage values - do not estimate
- Explain PoE class implications if relevant (Class 3 = 15.4W available, Class 4 = 30W, etc.)
- If calculating totals, show your work using the verified values
- Note any cameras missing power data"""

ACCESSORY_QUERY_TEMPLATE = """The engineer needs accessory/mounting information.

CONTEXT FROM DOCUMENTATION:
{context}

QUESTION: {question}

Instructions:
- List compatible accessories with their part numbers
- Note any mounting requirements or limitations
- If images are referenced, mention them for visual verification
- Clarify if accessories are sold separately or included"""

COMPARISON_TEMPLATE = """The engineer wants to compare camera options.

CONTEXT FROM DOCUMENTATION:
{context}

MODELS TO COMPARE: {models}

QUESTION: {question}

Create a comparison focusing on:
- Key specifications (resolution, sensor size, lens)
- Power requirements (from verified data)
- Environmental ratings (IP, IK, temperature)
- Unique features of each model

Format as a clear comparison, noting which specs come from which document."""

SPECIFICATION_TEMPLATE = """The engineer needs detailed specifications.

CONTEXT FROM DOCUMENTATION:
{context}

MODEL: {model}

QUESTION: {question}

Provide specifications in a structured format:
- Cite the exact document and page for each spec
- Group related specifications together
- Note if any requested specs are not in the documentation"""


# =============================================================================
# CONVERSATION TEMPLATES
# =============================================================================

FOLLOWUP_TEMPLATE = """Previous conversation:
{history}

New context (if any):
{context}

Follow-up question: {question}

Continue the conversation, referencing previous context when relevant."""

CLARIFICATION_TEMPLATE = """The question is ambiguous. Ask for clarification.

Original question: {question}

Possible interpretations:
{interpretations}

Ask the engineer to clarify which they meant."""


# =============================================================================
# RESPONSE FORMATTING
# =============================================================================

def format_context(results: list[dict], max_length: int = 4000) -> str:
    """
    Format search results into context string for prompt.

    Args:
        results: List of search results with 'content' and 'metadata'.
        max_length: Maximum context length in characters.

    Returns:
        Formatted context string.
    """
    context_parts = []
    current_length = 0

    for i, result in enumerate(results):
        content = result.get("content", "")
        metadata = result.get("metadata", {})

        source = metadata.get("source_file", "Unknown")
        page = metadata.get("page_num", "?")
        vendor = metadata.get("vendor", "").upper()
        model = metadata.get("model_num", "")

        # Format header
        header = f"[Source: {source}, Page {page}]"
        if vendor:
            header = f"[{vendor}] {header}"
        if model:
            header += f" (Model: {model})"

        formatted = f"{header}\n{content}\n"

        # Check length
        if current_length + len(formatted) > max_length:
            break

        context_parts.append(formatted)
        current_length += len(formatted)

    return "\n---\n".join(context_parts)


def format_poe_data(poe_info: dict) -> str:
    """
    Format POE metadata for prompt injection.

    Args:
        poe_info: Dict with 'by_model', 'total_watts', 'missing' keys.

    Returns:
        Formatted POE data string.
    """
    lines = []

    if poe_info.get("by_model"):
        lines.append("Power Consumption by Model:")
        for model, watts in poe_info["by_model"].items():
            lines.append(f"  - {model}: {watts}W")

        if len(poe_info["by_model"]) > 1:
            lines.append(f"  TOTAL: {poe_info['total_watts']:.1f}W")

    if poe_info.get("missing"):
        lines.append(f"\nMissing data for: {', '.join(poe_info['missing'])}")

    return "\n".join(lines) if lines else "No POE data available."


def format_metadata_summary(results: list[dict]) -> str:
    """
    Extract and format key metadata from search results.

    Args:
        results: List of search results with metadata.

    Returns:
        Formatted metadata summary.
    """
    lines = []
    seen_models = set()

    for result in results:
        meta = result.get("metadata", {})
        model = meta.get("model_num")

        if model and model not in seen_models:
            seen_models.add(model)
            parts = [f"Model: {model}"]

            if meta.get("poe_wattage"):
                parts.append(f"Power: {meta['poe_wattage']}W")
            if meta.get("poe_class"):
                parts.append(f"PoE Class {meta['poe_class']}")
            if meta.get("resolution"):
                parts.append(f"Resolution: {meta['resolution']}")
            if meta.get("ip_rating"):
                parts.append(f"{meta['ip_rating']}")

            lines.append(" | ".join(parts))

    return "\n".join(lines) if lines else "No structured metadata extracted."


# =============================================================================
# QUERY CLASSIFICATION
# =============================================================================

QUERY_TYPES = {
    "poe": ["power", "watt", "poe", "consumption", "budget", "draw"],
    "accessory": ["mount", "bracket", "pendant", "accessory", "compatible", "fit"],
    "comparison": ["compare", "versus", "vs", "difference", "better", "choose"],
    "specification": ["spec", "resolution", "sensor", "lens", "fps", "temperature", "ip rating"],
}


def classify_query(query: str) -> str:
    """
    Classify query type to select appropriate prompt template.

    Args:
        query: User's question.

    Returns:
        Query type: 'poe', 'accessory', 'comparison', 'specification', or 'general'.
    """
    query_lower = query.lower()

    scores = {}
    for query_type, keywords in QUERY_TYPES.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[query_type] = score

    if scores:
        return max(scores, key=scores.get)

    return "general"
