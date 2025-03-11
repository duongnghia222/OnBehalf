#!/usr/bin/env python3
"""
Chat with GPT-4o-mini with custom tools including Vertex AI image generation.
"""

import os
import json
import base64
from typing import List, Dict, Any, Optional
import argparse
from dotenv import load_dotenv
import openai
from openai.types.chat import ChatCompletionMessageParam
from vertex_image_generator import VertexImageGenerator

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the image generation tool
IMAGE_GEN_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an image using Google Vertex AI based on a text prompt",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A detailed description of the image to generate"
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "Elements to avoid in the generated image",
                },
                "width": {
                    "type": "integer",
                    "description": "Width of the generated image (default: 1024)"
                },
                "height": {
                    "type": "integer",
                    "description": "Height of the generated image (default: 1024)"
                }
            },
            "required": ["prompt"]
        }
    }
}

class ChatSession:
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize a chat session with GPT model."""
        self.model = model
        self.messages: List[ChatCompletionMessageParam] = []
        self.image_generator = VertexImageGenerator()
        
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
    
    def handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process tool calls from the assistant."""
        tool_results = []
        
        print(f"Processing {len(tool_calls)} tool call(s)")
        
        for tool_call in tool_calls:
            print(f"Tool call: {tool_call.function.name}")
            
            if tool_call.function.name == "generate_image":
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Extract parameters with defaults
                    prompt = function_args.get("prompt", "")
                    negative_prompt = function_args.get("negative_prompt", "")
                    width = function_args.get("width", 1024)
                    height = function_args.get("height", 1024)
                    
                    print(f"Generating image with prompt: '{prompt}'")
                    print(f"Negative prompt: '{negative_prompt}'")
                    print(f"Dimensions: {width}x{height}")
                    
                    # Call the image generator
                    image_path = self.image_generator.generate_image(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height
                    )
                    
                    print(f"Image generated at: {image_path}")
                    
                    # Add the tool response to messages
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "generate_image",
                        "content": json.dumps({"image_path": image_path})
                    })
                except Exception as e:
                    print(f"Error processing image generation tool call: {e}")
                    # Return an error message as the tool result
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "generate_image",
                        "content": json.dumps({"error": str(e)})
                    })
            else:
                print(f"Unknown tool: {tool_call.function.name}")
                
        return tool_results
    
    def chat(self, user_input: str) -> str:
        """Process user input, get response from GPT, and handle any tool calls."""
        # Add user message to history
        self.add_message("user", user_input)
        
        # Get response from GPT
        response = openai.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=[IMAGE_GEN_TOOL],
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        self.messages.append(assistant_message)
        
        # Check if the model wants to use tools
        if assistant_message.tool_calls:
            # Process tool calls
            tool_results = self.handle_tool_calls(assistant_message.tool_calls)
            
            # Add tool results to messages
            for tool_result in tool_results:
                self.messages.append(tool_result)
            
            # Get a new response from the model with the tool results
            second_response = openai.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            
            final_response = second_response.choices[0].message
            self.messages.append(final_response)
            return final_response.content or ""
        
        return assistant_message.content or ""

def main():
    """Main function to run the chat application."""
    parser = argparse.ArgumentParser(description="Chat with GPT-4o-mini with custom tools")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    args = parser.parse_args()
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file with your OpenAI API key or set it in your environment.")
        return
    
    # Initialize chat session
    chat_session = ChatSession(model=args.model)
    
    print(f"Chat session initialized with {args.model}. Type 'exit' to quit.")
    print("You can ask for image generation and the assistant will use Vertex AI to create images.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        print("\nAssistant: ", end="")
        response = chat_session.chat(user_input)
        print(response)

if __name__ == "__main__":
    main() 