# Big Data Migrator

An intelligent system for processing, analyzing, and migrating large data files with LLM-powered conversation capabilities.

## Features

- **Large File Processing**: Handle large CSV, Excel, PDF, and other file formats with memory-efficient processing
- **Data Context Understanding**: Intelligent analysis of data files including statistics and relationships
- **LLM-powered Conversations**: Discuss your data with both local and online LLMs
- **Memory Optimization**: Smart resource management for processing large datasets
- **User Guidance**: Get intelligent suggestions and recommendations for data exploration
- **Schema Optimization**: Improve data structures with advanced relationship detection

## Components

### Data Processing

- **Base Processors**: Framework for handling different file formats
- **Memory Monitoring**: Track and optimize memory usage during processing
- **Multi-file Processing**: Process and relate multiple data files

### LLM Conversation System

- **LMStudioClient**: Integration with locally-hosted LLMs like CodeLlama-34B
- **ConversationManager**: Maintain data-aware conversations with history
- **DataContextBuilder**: Create intelligent data summaries and insights
- **UserGuidanceSystem**: Generate guidance and recommendations based on data
- **OnlineLLMFallback**: Fallback to powerful online LLMs for complex tasks

## Getting Started

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/big-data-migrator.git
   cd big-data-migrator
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```

4. Create configuration:
   ```
   cp config/.env.example config/.env
   # Edit config/.env with your settings
   ```

### Using the LLM Conversation System

#### Chat Interface

Use the Streamlit chat interface for an interactive data discussion experience:

```
python start_frontend.py
```

Then navigate to the "Data Chat" or "Data Explorer" pages in the sidebar.

#### CLI Tool

Use the included CLI tool for quick data conversations:

```
python llm_conversation_cli.py --data path/to/your/data.csv --title "Data Analysis"
```

Optional flags:
- `--online`: Enable online LLM fallback
- `--model`: Specify a local model name

#### In Python Code

```python
from app.llm.conversation_system import LLMConversationSystem

# Initialize the system
llm_system = LLMConversationSystem()

# Create a conversation with data files
conversation_id = llm_system.create_conversation(
    title="Data Analysis",
    data_files=["/path/to/data.csv"]
)

# Add a message and get response
response = llm_system.add_message(
    message="What insights can you find in this data?",
    conversation_id=conversation_id
)

print(response["response"])

# Generate guidance
guidance = llm_system.generate_guidance(conversation_id)
```

#### API Endpoints

The system exposes several API endpoints:

- `POST /llm/conversations`: Create a new conversation
- `GET /llm/conversations/{id}`: Get conversation details
- `POST /llm/conversations/{id}/messages`: Add a message and get response
- `POST /llm/conversations/{id}/guidance`: Generate guidance for conversation
- `POST /llm/conversations/{id}/optimize-schema`: Optimize schema using online LLM

### Demo Notebook

Check out the example Jupyter notebook for a complete walkthrough:

```
jupyter notebook notebooks/llm_conversation_demo.ipynb
```

## Requirements

- Python 3.8+
- FastAPI and Uvicorn for API
- LM Studio (or compatible API server) for local LLM hosting
- Pandas, NumPy, and other data science libraries
- OpenAI API key (optional, for online LLM fallback)

## Configuration

Key environment variables:

- `LOCAL_LLM_URL`: URL for local LLM (default: http://localhost:1234/v1)
- `LOCAL_LLM_MODEL`: Model name (default: CodeLlama-34B-Instruct)
- `ENABLE_ONLINE_FALLBACK`: Enable online fallback (true/false)
- `OPENAI_API_KEY`: API key for online fallback

## Documentation

- [LLM Conversation System](docs/llm_conversation_system.md): Detailed guide on the conversation system
- API Documentation: Available at `http://localhost:8000/docs` when the API is running

## License

MIT
