import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

REFUSAL_MESSAGE = (
    "I couldn't find enough information in the uploaded documents."
)




def call_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip()




def build_context(retrieved_chunks):
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"""
DOCUMENT: {chunk['document_name']}
PAGE: {chunk['page_number']}

{chunk['text']}
"""
        )

    return "\n---\n".join(
        context_parts
    )




def build_chat_history(
    messages,
    max_messages=6
):
    if not messages:
        return "No previous conversation."

    recent_messages = messages[
        -max_messages:
    ]

    history_parts = []

    for message in recent_messages:
        role = message["role"].upper()
        content = message["content"]

        history_text = (
            f"{role}: {content}"
        )

        if message["role"] == "assistant":
            supporting_documents = (
                message.get(
                    "supporting_documents",
                    []
                )
            )

            if supporting_documents:
                documents_text = ", ".join(
                    supporting_documents
                )

                history_text += (
                    "\nSUPPORTING DOCUMENTS: "
                    f"{documents_text}"
                )

        history_parts.append(
            history_text
        )

    return "\n\n".join(
        history_parts
    )




def detect_explicit_document(
    question,
    document_names
):
    question_lower = (
        question.lower()
    )

    sorted_document_names = sorted(
        document_names,
        key=len,
        reverse=True
    )

    for document_name in (
        sorted_document_names
    ):
        if (
            document_name.lower()
            in question_lower
        ):
            return document_name

    return None




def rewrite_question(
    question,
    messages,
    document_names
):
    explicit_document = (
        detect_explicit_document(
            question,
            document_names
        )
    )

    
    if explicit_document is not None:
        return question

    chat_history = build_chat_history(
        messages
    )

    document_list = "\n".join(
        f"- {name}"
        for name in document_names
    )

    prompt = f"""
You rewrite follow-up questions into standalone questions
for document retrieval.

AVAILABLE DOCUMENTS:
{document_list}

RECENT CONVERSATION:
{chat_history}

CURRENT USER QUESTION:
{question}

Rules:

- Do NOT answer the question.
- Return only a standalone version of the user's question.
- Preserve the user's exact intent.
- Use previous conversation context only when necessary.
- Pay attention to SUPPORTING DOCUMENTS from previous answers.
- Never change an explicit document name supplied by the user.
- Never substitute one available document for another.
- Do not invent facts.

Resolve references such as:
- "it"
- "that"
- "this"
- "he"
- "she"
- "they"
- "the other document"
- "the other resume"
- "the first document"
- "the second document"

If the previous answer was supported by one document and
the user asks about "the other resume" or "the other document",
identify another available document that is not the previous
supporting document.

If the question is already standalone,
return it unchanged.

Return ONLY the rewritten question.

STANDALONE QUESTION:
"""

    rewritten = call_ollama(
        prompt
    )

    if not rewritten:
        return question

    return rewritten




def generate_answer(
    question,
    retrieved_chunks
):
    if not retrieved_chunks:
        return REFUSAL_MESSAGE

    context = build_context(
        retrieved_chunks
    )

    prompt = f"""
You are DocuMind, a document question-answering assistant.

Answer the QUESTION using ONLY the DOCUMENT CONTEXT below.

Rules:

- Read the document context carefully.
- If the answer is explicitly stated anywhere in the context,
  answer using that information.
- Evidence from one retrieved passage is enough.
- Ignore retrieved passages that are irrelevant to the question.
- Do not use outside knowledge.
- Do not guess or invent missing information.
- Do not infer that something is true merely because a related
  fact appears in the context.
- Do not infer that something is false merely because it is
  absent from the context.
- The person or entity in the evidence must match the person or
  entity asked about in the question.
- Preserve exact names, numbers, percentages, dates,
  certification names, job titles and technology names.
- Preserve category distinctions.
- For list questions, return the relevant items explicitly stated
  in the context.
- Keep the answer concise.

If the question cannot be answered directly from the document
context, return EXACTLY this sentence and nothing else:

{REFUSAL_MESSAGE}

DOCUMENT CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""

    answer = call_ollama(
        prompt
    )

    if not answer:
        return REFUSAL_MESSAGE

    return answer