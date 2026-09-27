import re

import numpy as np

from src.embeddings import get_embedding_model


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at",
    "be", "by", "does", "for", "from",
    "has", "have", "he", "her", "his",
    "how", "i", "in", "is", "it", "know",
    "mentioned", "of", "on", "or", "she",
    "that", "the", "their", "they", "this",
    "to", "was", "what", "when", "where",
    "which", "who", "with"
}


def cosine_similarity(
    vector_a,
    vector_b
):
    dot_product = np.dot(
        vector_a,
        vector_b
    )

    magnitude_a = np.linalg.norm(
        vector_a
    )

    magnitude_b = np.linalg.norm(
        vector_b
    )

    if (
        magnitude_a == 0
        or magnitude_b == 0
    ):
        return 0.0

    return (
        dot_product
        / (
            magnitude_a
            * magnitude_b
        )
    )



# Text tokenization


def tokenize(text):
    words = re.findall(
        r"[a-zA-Z0-9+#.-]+",
        text.lower()
    )

    return [
        word
        for word in words
        if (
            word not in STOP_WORDS
            and len(word) > 1
        )
    ]




def keyword_score(
    question,
    chunk_text
):
    question_words = set(
        tokenize(question)
    )

    if not question_words:
        return 0.0

    chunk_words = set(
        tokenize(chunk_text)
    )

    matches = (
        question_words
        & chunk_words
    )

    return (
        len(matches)
        / len(question_words)
    )




def detect_mentioned_document(
    original_question,
    chunks
):
    question_lower = (
        original_question.lower()
    )

    document_names = []

    for chunk in chunks:

        document_name = (
            chunk["document_name"]
        )

        if document_name not in document_names:
            document_names.append(
                document_name
            )

    # Check longer names first
    document_names.sort(
        key=len,
        reverse=True
    )

    for document_name in document_names:

        if (
            document_name.lower()
            in question_lower
        ):
            return document_name

    return None




def search_chunks(
    question,
    chunks,
    embeddings,
    top_k=3,
    original_question=None
):
    model = get_embedding_model()

    question_embedding = model.encode(
        question
    )

    
    if original_question is None:
        original_question = question


    
    # Detect document from the
    # ORIGINAL user question
    

    mentioned_document = (
        detect_mentioned_document(
            original_question,
            chunks
        )
    )


   
    # Score chunks
    

    results = []

    for (
        chunk,
        chunk_embedding
    ) in zip(
        chunks,
        embeddings
    ):

        
        if (
            mentioned_document
            is not None
            and chunk["document_name"]
            != mentioned_document
        ):
            continue


        semantic_score = (
            cosine_similarity(
                question_embedding,
                chunk_embedding
            )
        )

        lexical_score = (
            keyword_score(
                question,
                chunk["text"]
            )
        )


        

        hybrid_score = (
            0.75 * semantic_score
            + 0.25 * lexical_score
        )


        results.append({
            "text":
                chunk["text"],

            "page_number":
                chunk["page_number"],

            "document_name":
                chunk["document_name"],

            "score":
                float(hybrid_score),

            "semantic_score":
                float(semantic_score),

            "keyword_score":
                float(lexical_score),

            "hybrid_score":
                float(hybrid_score)
        })


    results.sort(
        key=lambda result:
            result["hybrid_score"],
        reverse=True
    )

    return results[:top_k]