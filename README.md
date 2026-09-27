# DocuMind — AI Document Assistant

DocuMind is a local AI document assistant I built to explore how Retrieval-Augmented Generation (RAG) can be used to answer questions across multiple PDF documents.

Instead of asking a language model to answer from general knowledge, DocuMind first searches the uploaded documents for relevant information and then uses that evidence to generate a grounded answer. The supporting document and page are also shown so the user can inspect where the answer came from.

The application runs locally using Ollama and does not require a hosted LLM API.

## Features

- Upload and search multiple PDF documents
- Ask questions through a Streamlit chat interface
- Hybrid retrieval using semantic similarity and keyword matching
- Ask document-specific questions by mentioning a PDF filename
- Handle conversational follow-up questions
- Generate answers grounded in retrieved document context
- Show supporting document names, page numbers, and retrieved passages
- Refuse unsupported questions instead of guessing
- Run the language model locally through Ollama
- Inspect retrieval details through developer information

## How It Works

DocuMind follows a Retrieval-Augmented Generation pipeline:

1. Uploaded PDFs are read page by page using `pypdf`.
2. Extracted text is divided into structure-aware chunks.
3. Each chunk is converted into an embedding using `all-MiniLM-L6-v2`.
4. When a question is asked, DocuMind creates an embedding for the question.
5. Relevant chunks are ranked using semantic similarity and keyword overlap.
6. The highest-ranked chunks are passed to `qwen2.5:7b` through Ollama.
7. The model is instructed to answer only from the retrieved document context.
8. Supporting document and page information is displayed with the answer.

The hybrid retrieval score currently uses:

```text
75% semantic similarity + 25% keyword similarity
```

If the user explicitly mentions an uploaded PDF filename, retrieval is restricted to that document.

For conversational follow-up questions, DocuMind can use recent conversation context to rewrite the question into a standalone retrieval query.

## Tech Stack

- Python
- Streamlit
- pypdf
- Sentence Transformers
- `all-MiniLM-L6-v2`
- NumPy
- Ollama
- `qwen2.5:7b`
- Requests

## Project Structure

```text
documind-ai/
├── src/
│   ├── __init__.py
│   ├── chunker.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── rag.py
│   └── vector_store.py
├── data/
│   └── readme.md
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── .env.example
```

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd documind-ai
```

Replace `<YOUR-GITHUB-REPOSITORY-URL>` with the URL of this repository.

### 2. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama from:

https://ollama.com/

Then download the model used by DocuMind:

```bash
ollama pull qwen2.5:7b
```

Make sure Ollama is running before starting the application.

### 5. Run DocuMind

```bash
streamlit run app.py
```

Streamlit will display a local address in the terminal. Open it in your browser, upload one or more PDF documents from the sidebar, and start asking questions.

## Example Questions

After uploading documents, questions can look like:

```text
What job title is mentioned in resume.pdf?

What technologies were used in the fraud detection project?

What certifications are listed?

What databases are mentioned?

What accuracy was achieved in the classification project?
```

Mentioning the exact filename can be useful when several uploaded PDFs contain similar information.

## Retrieval Approach

One of the main areas I focused on while building DocuMind was retrieval quality.

Semantic search is useful for finding passages that are conceptually related to a question, but exact terms, short skills, numbers, and resume-style information can also be important. Because of this, I combined semantic similarity with keyword matching instead of relying on semantic retrieval alone.

The current hybrid score is:

```text
hybrid_score = 0.75 × semantic_score + 0.25 × keyword_score
```

The application also checks the original user question for an explicit document filename. When one is present, chunks from other PDFs are excluded from the candidate results.

## Grounded Answer Generation

Retrieved passages are passed to the local Qwen model as document context.

The generation instructions require the model to:

- use only the retrieved document context
- ignore irrelevant retrieved passages
- avoid outside knowledge
- avoid guessing missing information
- preserve exact names, numbers, percentages, job titles, certifications, and technology names
- keep entity information separate
- return a fixed refusal response when the documents do not contain enough evidence

This was important because a useful document assistant should be able to say that the available evidence is insufficient rather than inventing an answer.

## Evaluation

I created a 15-question regression test set to check the final RAG pipeline.

The test set included:

- direct factual questions
- skills and technology questions
- list-based questions
- numerical information
- document-specific retrieval
- unsupported skills and experience
- unsupported research information
- entity mismatch cases

The final version passed **15/15 questions in this defined evaluation set**.

This is a project-specific regression test and should not be interpreted as a claim of universal RAG accuracy.

## Privacy

DocuMind is designed around local processing and local LLM inference.

PDF extraction, embedding generation, retrieval, and Ollama-based generation can run on the user's machine.

Uploaded test documents, private PDFs, credentials, `.env` files, and other sensitive information should not be committed to the repository.

## Current Limitations

- PDFs must contain extractable text; OCR for scanned or image-only PDFs is not currently implemented.
- Document embeddings are processed in memory rather than stored in a persistent vector database.
- The current version is configured for `qwen2.5:7b` through Ollama.
- Retrieval quality can vary depending on the structure and wording of the source documents.
- The current evaluation set is intentionally small and project-specific.

## Future Improvements

Some areas I would like to explore next include:

- OCR support for scanned PDFs
- persistent vector storage
- larger automated RAG evaluation sets
- configurable embedding models
- configurable local or hosted LLM providers
- additional document metadata filters
- deployment-oriented configuration

## Author

**Anam Intizar**

Data Science / AI Portfolio Project
