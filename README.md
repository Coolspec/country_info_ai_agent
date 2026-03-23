# Country Information AI Agent

## Problem Statement
This project implements an AI agent that answers user questions about countries using public data.

Example user questions:
- “What is the population of Germany?”
- “What currency does Japan use?”
- “What is the capital and population of Brazil?”

## Data Source
The agent retrieves real-time country data from the public REST Countries API:
[https://restcountries.com/v3.1/name/{country}](https://restcountries.com/v3.1/name/{country})

## Technical Architecture

The AI Agent is built with **LangGraph** to support a multi-step reasoning process, ensuring robust and accurate answers. The agent pipeline includes:
1. **Intent / Field Identification:** Determines what information the user is asking for.
2. **Tool Invocation:** Calls the REST Countries API to fetch the required data.
3. **Answer Synthesis:** Formats the retrieved data into a clear, natural language response.

## Constraints & Requirements
- **No authentication** required.
- **No database** is used; all data is fetched live.
- **No embeddings or RAG**; relies entirely on the tool invocation via the REST API.
- Designed as a **production-ready service**, handling invalid inputs and partial data gracefully.

---

## Project Structure

The project is divided into two main components:
- `backend/`: A FastAPI application that serves the LangGraph AI agent.
- `frontend/`: A Streamlit web interface for interacting with the agent.

---

## Getting Started

### Prerequisites
- Python 3.10+
- An API key for the LLM provider (e.g., OpenAI, Google GenAI, Groq) configured in the backend environment.

### 1. Running the Backend

The backend is a FastAPI server that exposes the LangGraph agent via a REST endpoint.

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your environment variables:
   Create a `.env` file in the `backend` directory and add your LLM API keys (e.g., `OPENAI_API_KEY`, `GROQ_API_KEY`, etc.).

5. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will start running at `http://localhost:8000`.

### 2. Running the Frontend

The frontend is a Streamlit application that provides a ChatGPT-like UI to chat with the agent.

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   The frontend will automatically open in your browser, running by default at `http://localhost:8501`. It will connect to the backend at `http://localhost:8000`. If your backend is running elsewhere, you can configure the API Base URL in the Streamlit sidebar.
