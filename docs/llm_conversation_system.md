# Intelligent LLM Conversation System

This component of the Big Data Migrator project provides data-aware conversations with local LLMs, with optional fallback to online models for complex tasks.

## Features

- **Local LLM Integration** - Connect to powerful locally hosted LLMs like CodeLlama-34B for privacy and cost efficiency
- **Conversation Management** - Track multi-turn conversations with data context awareness
- **Data Context Building** - Analyze data files to extract insights and relationships
- **Intelligent Guidance** - Receive suggestions and recommendations based on your data
- **Online LLM Fallback** - Leverage cloud-based LLMs for complex schema optimization tasks

## Components

1. **LMStudioClient**
   - Connects to local CodeLlama-34B model
   - Handles token counting and context optimization
   - Provides both streaming and standard responses

2. **ConversationManager**
   - Manages conversation history with data context
   - Compresses long conversations for memory efficiency
   - Persists conversations to disk

3. **DataContextBuilder**
   - Analyzes data files to build intelligent summaries
   - Extracts statistics and sample data
   - Detects relationships between data files

4. **UserGuidanceSystem**
   - Generates intelligent questions about data
   - Provides exploration suggestions
   - Offers schema optimization recommendations

5. **OnlineLLMFallback**
   - Handles complex schema optimization
   - Performs advanced relationship detection
   - Falls back to online LLMs when needed

## API Endpoints

### Conversations

- `POST /llm/conversations` - Create a new conversation
- `GET /llm/conversations/{id}` - Get conversation details

### Messages

- `POST /llm/conversations/{id}/messages` - Add a message and get response

### Guidance

- `POST /llm/conversations/{id}/guidance` - Generate guidance for conversation

### Schema Optimization

- `POST /llm/conversations/{id}/optimize-schema` - Optimize schema using online LLM

## Configuration

Configure the system via environment variables:

```
# Required
LOCAL_LLM_URL=http://localhost:1234/v1
LOCAL_LLM_MODEL=CodeLlama-34B-Instruct

# Optional
ENABLE_ONLINE_FALLBACK=false  # Set to true to enable online fallback
OPENAI_API_KEY=your_api_key   # Required if ENABLE_ONLINE_FALLBACK is true
ONLINE_LLM_MODEL=gpt-4o       # Online model to use for fallback
CONVERSATION_DIR=conversations # Directory to store conversations
```

## Example Usage

```python
# Initialize the conversation system
from app.llm.conversation_system import LLMConversationSystem

conversation_system = LLMConversationSystem()

# Create a conversation with data files
data_files = ['/path/to/sales.csv', '/path/to/customers.xlsx']
conversation_id = conversation_system.create_conversation(
    title="Sales Analysis",
    data_files=data_files
)

# Add a user message and get response
response = conversation_system.add_message(
    message="What are the top selling products?",
    conversation_id=conversation_id
)

# Generate guidance
guidance = conversation_system.generate_guidance(conversation_id)

# For complex schema tasks, use online fallback
if conversation_system.enable_online_fallback:
    schema = conversation_system.optimize_schema_with_fallback(conversation_id)
```

## Memory Management

The system is designed to be memory-efficient:

- Token counting to prevent context window overflow
- Conversation compression for long discussions
- Memory monitoring during data context building
- Resource optimization for large data files
