# AI ThinkTank
AI ThinkTank can be accessed [here](https://thinktank.streamlit.app/).

AI ThinkTank is an advanced Multi-Agent Debate platform that leverages Large Language Models to simulate expert debates on any topic. 

## Features
- **Image-to-Topic Generation**: Users can input an image URL, and the system automatically generates a debate topic based on the image using Gemini Pro Vision.
- **Dynamic Persona Generation**: The system dynamically spawns diverse expert personas based on the topic.
- **RAG-Enabled Debates**: Upload PDF documents to inject domain-specific knowledge into the debate. Experts will dynamically query the document vector store to support their arguments with retrieved facts.
- **Interactive Discussion**: Users can participate directly in the debate, interacting with the AI experts in real-time.
- **LangSmith Tracing**: Integrated LLM chain tracing and performance debugging.

---

## Technical Stack & Architecture
- **Framework**: LangChain & LangGraph (for multi-agent flow orchestration)
- **Frontend**: Streamlit
- **Model**: Gemini Pro 1.0 & Gemini Pro Vision
- **RAG Pipeline**:
  - **Document Parsing**: `PyPDF2` for text extraction.
  - **Text Splitter**: `RecursiveCharacterTextSplitter` (chunk size: 10,000, overlap: 1,000).
  - **Embeddings**: Google AI `embedding-001` or Hugging Face `sentence-transformers` models (configured in code).
  - **Vector Database**: FAISS (Facebook AI Similarity Search) for local vector storage and similarity search.
  - **Retrieval**: Context-aware similarity search mapping relevant facts back to the agents' system prompts to ground discussions and reduce hallucinations.
- **Observability**: **LangSmith** for tracing, debugging, and evaluation.

---

## Getting Started

1. **Install Dependencies**
   Navigate to the cloned repository directory and install the required Python packages:
   
   ```bash
   pip install -r requirements.txt
   ```
   
2. **Setup LangSmith (Optional, for LLM tracing)**
   Set up your environment variables to enable LangSmith debugging:
   
   ```bash
   export LANGCHAIN_TRACING_V2="true"
   export LANGCHAIN_API_KEY="your-langsmith-api-key"
   export LANGCHAIN_PROJECT="nlp-ai-thinktank"
   ```

3. **Run the Application**
   To start the application, run:    
   
   ```bash
   streamlit run Introduction.py
   ```
   
   This will launch the Streamlit web server locally.

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- AI ThinkTank was inspired by the [MAD Framework](https://github.com/Skytliang/Multi-Agents-Debate).
- The project makes use of the [LangChain Framework](https://www.langchain.com/).
