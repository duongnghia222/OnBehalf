import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Simplified conversation history - we'll rebuild it properly for each API call
        self.messages = []  # Store all messages in a simple format

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        print("server_script_path", server_script_path)
        
        # Convert relative path to absolute path
        if not os.path.isabs(server_script_path):
            # If path starts with .. or similar, resolve it relative to current directory
            server_script_path = os.path.abspath(os.path.join(os.getcwd(), server_script_path))
        
        print("Resolved path:", server_script_path)
        
        if not os.path.exists(server_script_path):
            raise FileNotFoundError(f"Server script not found: {server_script_path}")
            
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available tools"""
        # Add the new query to messages
        self.messages.append({"role": "user", "content": query})
        
        # Build a clean conversation history for the API
        api_messages = self._build_api_messages()
        
        response = await self.session.list_tools()
        available_tools = [{ 
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # Initial OpenAI API call
        final_text = []
        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=100,
                messages=api_messages,
                tools=available_tools,
                tool_choice="auto"
            )
            
            # Process response and handle tool calls
            assistant_message = response.choices[0].message
            
            # Store the assistant's response
            assistant_data = {
                "role": "assistant",
                "content": assistant_message.content or ""
            }
            
            # If there are tool calls, add them to the message
            if assistant_message.tool_calls:
                assistant_data["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in assistant_message.tool_calls
                ]
            
            self.messages.append(assistant_data)
            
            if assistant_message.content:
                final_text.append(assistant_message.content)
            
            # Check if the model wants to use tools
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    tool_result_text = f"[Calling tool {tool_name} with args {tool_args}]"
                    final_text.append(tool_result_text)
                    
                    # Add tool response to messages
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": result.content
                    })
                
                # Rebuild API messages with the new tool responses
                api_messages = self._build_api_messages()
                
                # Get next response from OpenAI with tool results
                response = self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=api_messages
                )
                
                final_response = response.choices[0].message
                
                # Add final response to messages
                self.messages.append({
                    "role": "assistant",
                    "content": final_response.content or ""
                })
                
                if final_response.content:
                    final_text.append(final_response.content)
                    
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            final_text.append(error_msg)
            # Print the API messages for debugging
            print("\nAPI Messages that caused the error:")
            for i, msg in enumerate(api_messages):
                print(f"{i}: {msg}")

        return "\n".join(final_text)
        
    def _build_api_messages(self):
        """Build a properly structured message list for the OpenAI API"""
        api_messages = []
        pending_tool_calls = {}  # Map of tool_call_id to whether it has a response
        
        for msg in self.messages:
            role = msg.get("role")
            
            if role == "user" or role == "system":
                # User and system messages can be added directly
                api_messages.append(msg.copy())
            
            elif role == "assistant":
                # For assistant messages, we need to track any tool calls
                assistant_msg = msg.copy()
                api_messages.append(assistant_msg)
                
                # If this assistant message has tool calls, mark them as pending
                if "tool_calls" in assistant_msg:
                    for tool_call in assistant_msg["tool_calls"]:
                        pending_tool_calls[tool_call["id"]] = False
            
            elif role == "tool":
                # For tool messages, add them only if they respond to a pending tool call
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id in pending_tool_calls:
                    api_messages.append(msg.copy())
                    pending_tool_calls[tool_call_id] = True
        
        # Check if all tool calls have responses
        missing_responses = [id for id, has_response in pending_tool_calls.items() if not has_response]
        if missing_responses:
            print(f"Warning: Missing tool responses for: {missing_responses}")
            # Remove assistant messages with tool calls that don't have responses
            api_messages = [msg for msg in api_messages if 
                           msg.get("role") != "assistant" or 
                           "tool_calls" not in msg or
                           not any(tc["id"] in missing_responses for tc in msg["tool_calls"])]
        
        return api_messages

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        print("Type 'clear' to clear conversation history.")
        print("Type 'history' to view the current conversation history.")
        print("Type 'debug' to view the raw message structure.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                
                if query.lower() == 'clear':
                    self.messages = []
                    print("Conversation history cleared.")
                    continue
                
                if query.lower() == 'history':
                    print("\n=== Conversation History ===")
                    for i, msg in enumerate(self.messages):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        
                        if role == "user":
                            print(f"\n[User {i}]: {content}")
                        elif role == "assistant":
                            print(f"\n[Assistant {i}]: {content}")
                            if "tool_calls" in msg:
                                tool_calls = msg["tool_calls"]
                                print(f"  [Tool Calls]: {len(tool_calls)} calls")
                                for tc in tool_calls:
                                    print(f"    - {tc['function']['name']} (ID: {tc['id']})")
                        elif role == "tool":
                            print(f"\n[Tool {i}]: {msg.get('name', 'unknown')} -> {content[:50]}...")
                    print("\n===========================")
                    continue
                
                if query.lower() == 'debug':
                    print("\n=== Raw Message Structure ===")
                    for i, msg in enumerate(self.messages):
                        print(f"\n[Message {i}]:")
                        print(json.dumps(msg, indent=2))
                    print("\n===========================")
                    continue
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
                import traceback
                traceback.print_exc()
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    import json
    asyncio.run(main())