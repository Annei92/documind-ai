# DocuMind — AI Document Assistant

DocuMind is a project I built to learn and experiment with Retrieval-Augmented Generation (RAG).

The idea came from a simple problem: when working with multiple documents, finding a specific piece of information can take time. I wanted to build something where I could upload a few PDFs, ask questions in normal language, and get answers based specifically on those documents.

DocuMind does exactly that. It searches the uploaded PDFs for relevant information, sends the most useful context to a local language model, and generates an answer. It also shows the source document and page so the answer can be checked instead of simply trusted.

Everything runs locally using Ollama.

## What DocuMind can do

- Upload multiple PDF documents
- Ask questions about them through a chat interface
- Search across all uploaded documents
- Search a particular PDF by mentioning its filename
- Understand follow-up questions in a conversation
- Show the document and page behind an answer
- Avoid answering when there isn't enough evidence in the documents
- Run the LLM locally instead of depending on a paid API

## How I built it

The application follows a RAG pipeline.

When PDFs are uploaded, I first extract their text page by page and divide it into smaller chunks.

Those chunks are converted into embeddings using `all-MiniLM-L6-v2`.

When the user asks a question, DocuMind compares the question with the document chunks to find the most relevant information.

I found that semantic similarity alone wasn't always ideal, especially for things like exact skills, technology names and numbers. To improve this, I combined semantic search with keyword matching.

The final retrieval score uses:

```text
75% semantic similarity + 25% keyword matching
