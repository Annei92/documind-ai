from io import BytesIO

import streamlit as st

from src.document_loader import extract_text_from_pdf
from src.chunker import chunk_pages
from src.embeddings import generate_embeddings
from src.vector_store import search_chunks
from src.rag import (
    generate_answer,
    rewrite_question
)


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide"
)

# =========================================================
# UI styling
# =========================================================

st.markdown(
    """
    <style>
        /* Main content width */
        .block-container {
            max-width: 1100px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        /* Slightly cleaner chat spacing */
        [data-testid="stChatMessage"] {
            padding-top: 0.8rem;
            padding-bottom: 0.8rem;
        }

        /* Sidebar spacing */
        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        /* Keep chat content comfortably readable */
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            line-height: 1.65;
        }

        /* Give source expanders a little breathing room */
        [data-testid="stExpander"] {
            margin-top: 0.45rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# =========================================================
# Main header
# =========================================================

st.title("📄 DocuMind")

st.caption(
    "AI-powered document intelligence with grounded answers "
    "and source citations."
)

st.markdown(
    "**Upload PDFs from the sidebar, then ask questions "
    "about your documents below.**"
)


# =========================================================
# Constants
# =========================================================

REFUSAL_MESSAGE = (
    "I couldn't find enough information in the uploaded documents."
)


def has_supporting_answer(answer):
    return answer.strip() != REFUSAL_MESSAGE


# =========================================================
# User-facing source display
# =========================================================

def display_sources(sources):
    """
    Display unique document/page sources to the user.

    Retrieval results themselves are NOT modified.
    This only removes duplicate document/page combinations
    from the user-facing source presentation.
    """

    unique_sources = []
    seen = set()

    for source in sources:

        source_key = (
            source["document_name"],
            source["page_number"]
        )

        if source_key not in seen:
            seen.add(source_key)
            unique_sources.append(source)

    if not unique_sources:
        return

    with st.expander(
        f"📚 View supporting sources ({len(unique_sources)})"
    ):

        for index, source in enumerate(
            unique_sources,
            start=1
        ):

            st.markdown(
                f"**{source['document_name']} "
                f"— Page {source['page_number']}**"
            )

            st.write(
                source["text"]
            )

            if index < len(unique_sources):
                st.divider()


# =========================================================
# Session state
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_signature" not in st.session_state:
    st.session_state.document_signature = None

if "last_rewritten_question" not in st.session_state:
    st.session_state.last_rewritten_question = None

if "last_results" not in st.session_state:
    st.session_state.last_results = []

if "last_error" not in st.session_state:
    st.session_state.last_error = None


# =========================================================
# Cached document processing
# =========================================================

@st.cache_data(show_spinner=False)
def process_documents(file_data):

    all_chunks = []
    document_info = []

    for file_name, file_bytes in file_data:

        pdf_file = BytesIO(
            file_bytes
        )

        pages = extract_text_from_pdf(
            pdf_file
        )

        chunks = chunk_pages(
            pages
        )

        for chunk in chunks:
            chunk["document_name"] = (
                file_name
            )

        all_chunks.extend(
            chunks
        )

        document_info.append({
            "name": file_name,
            "pages": len(pages),
            "chunks": len(chunks)
        })

    if not all_chunks:
        return (
            [],
            [],
            document_info
        )

    embeddings = generate_embeddings(
        all_chunks
    )

    return (
        all_chunks,
        embeddings,
        document_info
    )


# =========================================================
# Sidebar — document upload
# =========================================================

with st.sidebar:

    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )


# =========================================================
# Uploaded documents
# =========================================================

if uploaded_files:

    # -----------------------------------------------------
    # Prepare file data
    # -----------------------------------------------------

    file_data = []

    for file in uploaded_files:

        file_bytes = (
            file.getvalue()
        )

        file_data.append(
            (
                file.name,
                file_bytes
            )
        )


    # -----------------------------------------------------
    # Detect document changes
    # -----------------------------------------------------

    current_document_signature = tuple(
        (
            file_name,
            len(file_bytes),
            hash(file_bytes)
        )
        for (
            file_name,
            file_bytes
        ) in file_data
    )

    if (
        st.session_state.document_signature
        != current_document_signature
    ):

        st.session_state.messages = []

        st.session_state.last_rewritten_question = (
            None
        )

        st.session_state.last_results = []

        st.session_state.last_error = None

        st.session_state.document_signature = (
            current_document_signature
        )


    # -----------------------------------------------------
    # Process documents
    # -----------------------------------------------------

    with st.spinner(
        "Processing documents..."
    ):

        (
            all_chunks,
            embeddings,
            document_info
        ) = process_documents(
            file_data
        )


    document_names = [
        document["name"]
        for document in document_info
    ]


    # -----------------------------------------------------
    # Sidebar — document information
    # -----------------------------------------------------

    with st.sidebar:

        st.success(
            f"{len(document_info)} document(s) ready"
        )

        for document in document_info:

            with st.expander(
                f"📄 {document['name']}"
            ):

                st.write(
                    f"Pages: {document['pages']}"
                )

                st.write(
                    f"Chunks: {document['chunks']}"
                )

        st.caption(
            f"Total chunks: {len(all_chunks)}"
        )


    # -----------------------------------------------------
    # No readable content
    # -----------------------------------------------------

    if not all_chunks:

        st.warning(
            "No extractable text was found "
            "in the uploaded documents."
        )

        st.stop()


    # -----------------------------------------------------
    # Clear chat
    # -----------------------------------------------------

    if st.session_state.messages:

        if st.button(
            "Clear conversation",
            type="secondary",
            icon="🗑️",
            help="Clear the current conversation and start again."
        ):

            st.session_state.messages = []

            st.session_state.last_rewritten_question = (
                None
            )

            st.session_state.last_results = []

            st.session_state.last_error = None

            st.rerun()


    # -----------------------------------------------------
    # Empty chat state
    # -----------------------------------------------------

    if not st.session_state.messages:

        st.subheader(
            "Your documents are ready"
        )

        st.write(
            "Ask questions across your uploaded PDFs. "
            "DocuMind retrieves relevant passages and generates "
            "answers grounded in your documents."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown("### 🔎")

            st.markdown(
                "**Grounded Answers**"
            )

            st.caption(
                "Responses are generated from retrieved "
                "document content."
            )

        with col2:

            st.markdown("### 📚")

            st.markdown(
                "**Source Citations**"
            )

            st.caption(
                "Inspect the document and page supporting "
                "each answer."
            )

        with col3:

            st.markdown("### 📄")

            st.markdown(
                "**Multi-PDF Search**"
            )

            st.caption(
                "Search across multiple uploaded documents "
                "in one workspace."
            )

        st.info(
            "Tip: Mention a PDF filename when you want "
            "DocuMind to search a specific document."
        )


    # -----------------------------------------------------
    # Display chat history
    # -----------------------------------------------------

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and "sources" in message
            ):

                if has_supporting_answer(
                    message["content"]
                ):

                    display_sources(
                        message["sources"]
                    )

                else:

                    st.caption(
                        "No supporting sources found."
                    )


    # -----------------------------------------------------
    # Chat input
    # -----------------------------------------------------

    question = st.chat_input(
        "Ask anything about your uploaded documents..."
    )


    if question:

        # -------------------------------------------------
        # Save current user question
        # -------------------------------------------------

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })


        # -------------------------------------------------
        # Display user question
        # -------------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )


        try:

            # ---------------------------------------------
            # Rewrite question
            # ---------------------------------------------

            with st.spinner(
                "Understanding question..."
            ):

                previous_messages = (
                    st.session_state.messages[
                        :-1
                    ]
                )

                standalone_question = (
                    rewrite_question(
                        question,
                        previous_messages,
                        document_names
                    )
                )

                st.session_state.last_rewritten_question = (
                    standalone_question
                )


            # ---------------------------------------------
            # Retrieval
            # ---------------------------------------------

            with st.spinner(
                "Searching documents..."
            ):

                results = search_chunks(
                    standalone_question,
                    all_chunks,
                    embeddings,
                    top_k=3,
                    original_question=question
                )

                st.session_state.last_results = (
                    results
                )


            # ---------------------------------------------
            # Generate grounded answer
            # ---------------------------------------------

            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    standalone_question,
                    results
                )


            # ---------------------------------------------
            # Display assistant answer
            # ---------------------------------------------

            with st.chat_message(
                "assistant"
            ):

                st.write(
                    answer
                )

                if has_supporting_answer(
                    answer
                ):

                    display_sources(
                        results
                    )

                else:

                    st.caption(
                        "No supporting sources found."
                    )


            # ---------------------------------------------
            # Supporting document metadata
            # ---------------------------------------------

            supporting_documents = []

            if has_supporting_answer(
                answer
            ):

                for result in results:

                    document_name = (
                        result[
                            "document_name"
                        ]
                    )

                    if (
                        document_name
                        not in supporting_documents
                    ):

                        supporting_documents.append(
                            document_name
                        )


            # ---------------------------------------------
            # Save assistant response
            # ---------------------------------------------

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": results,
                "supporting_documents":
                    supporting_documents
            })

            st.session_state.last_error = None


        except Exception as error:

            error_message = (
                "DocuMind could not complete "
                "the request. Please make sure "
                "Ollama is running and try again."
            )

            with st.chat_message(
                "assistant"
            ):

                st.error(
                    error_message
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": [],
                "supporting_documents": []
            })

            st.session_state.last_error = (
                str(error)
            )


        st.rerun()


    # -----------------------------------------------------
    # Developer information
    # -----------------------------------------------------

    with st.expander(
        "Developer information"
    ):

        st.write(
            f"Chunks: {len(all_chunks)}"
        )

        if len(embeddings) > 0:

            st.write(
                "Embedding dimensions: "
                f"{len(embeddings[0])}"
            )


        # ---------------------------------------------
        # Rewritten question
        # ---------------------------------------------

        if (
            st.session_state
            .last_rewritten_question
        ):

            st.write(
                "Rewritten question:"
            )

            st.code(
                st.session_state
                .last_rewritten_question
            )


        # ---------------------------------------------
        # Retrieved results
        # ---------------------------------------------

        if (
            st.session_state
            .last_results
        ):

            st.write(
                "Retrieved documents:"
            )

            for (
                index,
                result
            ) in enumerate(
                st.session_state
                .last_results,
                start=1
            ):

                st.write(
                    f"{index}. "
                    f"{result['document_name']} "
                    f"— Page "
                    f"{result['page_number']} "
                    f"— Score "
                    f"{result['score']:.3f}"
                )


            # -----------------------------------------
            # Retrieved chunk text
            # -----------------------------------------

            st.write(
                "Retrieved chunk text:"
            )

            for (
                index,
                result
            ) in enumerate(
                st.session_state
                .last_results,
                start=1
            ):

                with st.expander(
                    f"Retrieved chunk {index}"
                ):

                    st.write(
                        f"Document: "
                        f"{result['document_name']}"
                    )

                    st.write(
                        f"Page: "
                        f"{result['page_number']}"
                    )

                    st.write(
                        f"Hybrid score: "
                        f"{result['score']:.3f}"
                    )

                    if (
                        "semantic_score"
                        in result
                    ):

                        st.write(
                            "Semantic score: "
                            f"{result['semantic_score']:.3f}"
                        )

                    if (
                        "keyword_score"
                        in result
                    ):

                        st.write(
                            "Keyword score: "
                            f"{result['keyword_score']:.3f}"
                        )

                    st.write(
                        "Chunk text:"
                    )

                    st.write(
                        result["text"]
                    )


        # ---------------------------------------------
        # Error debugging
        # ---------------------------------------------

        if (
            st.session_state
            .last_error
        ):

            st.write(
                "Last error:"
            )

            st.code(
                st.session_state
                .last_error
            )


# =========================================================
# No documents uploaded
# =========================================================

else:

    st.info(
        "Upload one or more PDF documents "
        "from the sidebar to begin."
    )