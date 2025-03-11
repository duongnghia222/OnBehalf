# GPT-4o-mini Chat with Vertex AI Image Generation

This project provides a command-line interface to chat with OpenAI's GPT-4o-mini model, enhanced with a custom tool for generating images using Google's Vertex AI.

## Features

- Interactive chat with GPT-4o-mini
- Custom tool for generating images using Google Vertex AI Imagen
- Support for both Vertex AI SDK and REST API implementations
- Conversation history management
- Error handling with visual feedback

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Google Cloud Project with Vertex AI API enabled
- Google Cloud service account with Vertex AI permissions

## Installation

1. Clone this repository or copy the files to your local machine.

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on the provided `.env.example`:
   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file and add your API keys and configuration:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_CLOUD_PROJECT=your_google_cloud_project_id
   GOOGLE_CLOUD_LOCATION=us-central1
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
   ```

## Usage

Run the chat script:

```bash
python chat_with_gpt.py
```

You can specify a different OpenAI model if needed:

```bash
python chat_with_gpt.py --model gpt-4o
```

### Example Conversation

```
Chat session initialized with gpt-4o-mini. Type 'exit' to quit.
You can ask for image generation and the assistant will use Vertex AI to create images.

You: Can you generate an image of a futuristic city with flying cars?