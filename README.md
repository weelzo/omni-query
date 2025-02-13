<div align="center">

  <!-- Logo Image -->
  <img src="docs/logo.png" alt="OmniQuery Logo" width="200"/>

  <!-- Title -->
  # OmniQuery

</div>

**OmniQuery** is a **Multimodal Retrieval-Augmented Generation (RAG)** system that allows users to upload PDFs containing text and images, ask questions about the content, and receive explanations with highlighted references to the relevant parts of the PDF. It combines text and image understanding to provide a seamless and interactive experience.

---

## Features

- **Multimodal Understanding**: Processes both text and images in PDFs.
- **Interactive Chat Interface**: Ask questions and get explanations with references.
- **Highlighting**: Highlights relevant text and image regions in the PDF.
- **OCR Integration**: Extracts text from images using Tesseract OCR.
- **Vector Search**: Uses FAISS for efficient text and image retrieval.
- **LLM Integration**: Generates responses using OpenAI's GPT-4 or other LLMs.

---

## Demo


---

## Installation

### Prerequisites

- Python 3.8 or higher
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/OmniQuery.git
   cd OmniQuery
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Project Structure
  ```bash
  omni-query/
  ├── app.py                  # Streamlit frontend
  ├── .env                    # Environment variables (API keys)
  ├── requirements.txt        # Dependencies
  ├── README.md               # Project documentation
  ├── modules/
  │   ├── __init__.py
  │   ├── pdf_processor.py    # PDF text and image extraction
  │   ├── ocr.py              # OCR for text in images
  │   ├── embeddings.py       # Text and image embeddings
  │   ├── vector_db.py        # FAISS vector database
  │   ├── llm.py              # LLM integration (e.g., OpenAI)
  │   └── utils.py            # Utility functions
  └── assets/                # For storing extracted images
  └── docs/
  ```
