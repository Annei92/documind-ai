# DocuMind — AI Document Assistant

DocuMind is a document question-answering application I built to explore how Retrieval-Augmented Generation (RAG) can be used to answer questions from multiple PDF documents.

The idea is simple: upload your PDFs, ask a question, and DocuMind searches the documents for the most relevant information before generating an answer. The response is grounded in the uploaded documents and includes the source document and page so the retrieved evidence can be checked.

The application runs locally using Ollama, so it does not depend on a hosted LLM API.

## What it can do

- Upload and search multiple PDF documents
- Ask questions in a chat-style interface
- Search documents using both semantic similarity and keyword matching
- Continue a conversation with follow-up questions
- Search a specific PDF by mentioning its filename
- Show the document and page used to support an answer
- Refuse questions when the uploaded documents do not contain enough information
- Run the language model locally with Ollama

## How it works

DocuMind uses a RAG pipeline:

1. PDF text is extracted page by page.
2. The extracted text is divided into smaller, structure-aware chunks.
3. Each chunk is converted into an embedding using `all-MiniLM-L6-v2`.
4. When a question is asked, DocuMind searches for the most relevant chunks.
5. Retrieval combines semantic similarity with keyword matching.
6. The best matching chunks are passed to a local `qwen2.5:7b` model through Ollama.
7. The model is instructed to answer only from the retrieved document context.
8. The relevant document and page are shown with the answer.

The retrieval score currently combines:

```text
75% semantic similarity + 25% keyword similarity
