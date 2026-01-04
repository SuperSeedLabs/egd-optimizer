import os
from typing import Dict, List, Optional, Tuple, Any, Union

import anthropic
import tiktoken
from dotenv import load_dotenv

load_dotenv()


class AnthropicLLM:
    """
    A wrapper for the Anthropic Claude API that manages prompts, token limits, and responses.
    
    This class handles system prompts, token counting, prompt truncation, and 
    response processing for the Anthropic Claude API.
    """
    
    def __init__(
            self, 
            system_prompt: str, 
            all_tools: List[Dict[str, Any]], 
            max_tokens: int = 200000, 
            temperature: float = 0.0,
            response_tokens: int = 1024
        ):
        """
        Initialize the Anthropic model with system prompt and available tools.
        
        Args:
            system_prompt: The system instructions to guide Claude's behavior
            all_tools: List of tool definitions in the format required by Claude's API
            max_tokens: Maximum tokens for the model's context window
            temperature: Controls randomness in output generation (0.0 to 1.0)
            response_tokens: Maximum number of tokens to generate in the response
        """
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC"))
        self.system_prompt = system_prompt
        self.all_tools = all_tools
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.response_tokens = response_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Class constants
        self.TOKEN_SAFETY_BUFFER = 100  # Buffer to prevent hitting token limits
        self.DEFAULT_MODEL = "claude-3-7-sonnet-20250219"

    def encode_text(self, text: str) -> List[int]:
        """
        Encode text into tokens using the Claude tokenizer.
        
        Args:
            text: The text to encode
            
        Returns:
            A list of token IDs
        """
        return self.tokenizer.encode(text, disallowed_special=())

    def decode_tokens(self, tokens: List[int]) -> str:
        """
        Decode tokens back into text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            The decoded text
        """
        return self.tokenizer.decode(tokens)

    def truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """
        Truncate a prompt to fit within the specified token limit.
        
        Args:
            prompt: The prompt to truncate
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Truncated prompt text
        """
        tokens = self.encode_text(prompt)
        if len(tokens) <= max_tokens:
            return prompt
        return self.decode_tokens(tokens[:max_tokens])

    def generate_response(self, prompt: str) -> Tuple[Optional[Union[str, Dict[str, Any]]], int, int, int]:
        """
        Generate a response from Claude based on the provided prompt.
        
        Args:
            prompt: The user prompt to send to Claude
            
        Returns:
            Tuple containing:
            - The response data (text or tool use input)
            - Total tokens used
            - Prompt tokens used
            - Response tokens used
        """
        # Calculate available tokens
        system_prompt_tokens = len(self.encode_text(self.system_prompt))
        available_tokens = self.max_tokens - system_prompt_tokens - self.TOKEN_SAFETY_BUFFER

        # Truncate prompt if needed
        truncated_prompt = self.truncate_prompt(prompt, available_tokens)
        prompt_tokens = len(self.encode_text(truncated_prompt))

        # Generate response from Claude
        response = self.anthropic_client.messages.create(
            model=self.DEFAULT_MODEL,
            messages=[
                {
                    "role": "user", 
                    "content": self.system_prompt + "\n" + prompt
                    },
            ],
            temperature=self.temperature,
            max_tokens=self.response_tokens,
            tools=self.all_tools,
        )

        # Process the response
        response_data, response_tokens = self.parse_response(response)
        total_tokens = system_prompt_tokens + prompt_tokens + response_tokens

        return response_data, total_tokens, prompt_tokens, response_tokens

    def parse_response(self, response: anthropic.types.Message) -> Tuple[Optional[Union[str, Dict[str, Any]]], int]:
        """
        Extract relevant data from Claude's response.
        
        Extracts either tool use data or text content from the response.
        
        Args:
            response: The response object from Claude API
            
        Returns:
            Tuple containing:
            - The extracted response data (tool input or text)
            - Number of tokens in the response
        """
        # Initialize default response values
        response_data = None
        num_tokens = 0

        # Verify content exists in the response
        if not hasattr(response, "content") or not response.content:
            print("No content found in the response.")
            return None, 0

        # First check for tool use blocks
        tool_use_blocks = [block for block in response.content if block.type == "tool_use"]

        if tool_use_blocks:
            # Process the first tool use block
            first_tool_use_block = tool_use_blocks[0]
            if hasattr(first_tool_use_block, "input") and first_tool_use_block.input:
                response_data = first_tool_use_block.input
            return response_data, num_tokens

        # If no tool use blocks, extract text from text blocks
        text_blocks = [block.text for block in response.content if block.type == "text"]
        if text_blocks:
            response_data = " ".join(text_blocks)
            print("Extracted text from text blocks:", response_data)
        else:
            print("No tool use blocks or text blocks found in the response.")

        return response_data, num_tokens
