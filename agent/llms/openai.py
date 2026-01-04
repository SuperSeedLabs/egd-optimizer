import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union

import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from agent.utils import count_tokens, anthropic_to_openai

load_dotenv()


class OpenAILLM:
    """
    A wrapper for the OpenAI API that manages prompts, token limits, and responses.
    
    This class handles system prompts, token counting, prompt truncation, and 
    response processing for the OpenAI API.
    """
    
    def __init__(
            self, 
            system_prompt: str, 
            all_tools: List[Dict[str, Any]], 
            max_tokens: int = 124000, 
            temperature: float = 0.0,
            response_tokens: int = 1024,
            model: str = "o3-mini"
        ):
        """
        Initialize the OpenAI model with system prompt and available tools.
        
        Args:
            system_prompt: The system instructions to guide the model's behavior
            all_tools: List of tool definitions in the format required by OpenAI API
            max_tokens: Maximum tokens for the model's context window
            temperature: Controls randomness in output generation (0.0 to 1.0)
            response_tokens: Maximum number of tokens to generate in the response
            model: The OpenAI model to use
        """
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI"))
        self.system_prompt = system_prompt
        self.all_tools = all_tools
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.response_tokens = response_tokens
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Class constants
        self.TOKEN_SAFETY_BUFFER = 100  # Buffer to prevent hitting token limits

    def encode_text(self, text: str) -> List[int]:
        """
        Encode text into tokens using the OpenAI tokenizer.
        
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
        Generate a response from the OpenAI model based on the provided prompt.
        
        Args:
            prompt: The user prompt to send to the model
            
        Returns:
            Tuple containing:
            - The response data (text or tool call arguments)
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

        # Generate response from OpenAI
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": truncated_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.response_tokens,
            tools=[anthropic_to_openai(tool) for tool in self.all_tools],
        )

        # Process the response
        response_data, response_tokens = self.parse_response(response)
        total_tokens = system_prompt_tokens + prompt_tokens + response_tokens

        return response_data, total_tokens, prompt_tokens, response_tokens

    def parse_response(self, response: Any) -> Tuple[Optional[Union[str, Dict[str, Any]]], int]:
        """
        Extract relevant data from the OpenAI response.
        
        Extracts either tool call arguments or text content from the response.
        
        Args:
            response: The response object from OpenAI API
            
        Returns:
            Tuple containing:
            - The extracted response data (tool call arguments or text)
            - Number of tokens in the response
        """
        # Initialize default response data
        response_data = None
        response_tokens = 0
        
        # Check if the response has 'choices' and if the first choice has 'tool_calls'
        if not hasattr(response, "choices") or not response.choices:
            print("No choices found in the response.")
            return None, 0
            
        for choice in response.choices:
            if hasattr(choice, "finish_reason") and choice.finish_reason == "tool_calls":
                # Extract tool call arguments
                response_data = choice.message.tool_calls[0].function.arguments
                response_tokens = count_tokens(response_data, "cl100k_base")
                # Parse the JSON string into a dictionary
                response_data = json.loads(response_data)
                return response_data, response_tokens
            else:
                # Extract text content
                response_data = choice.message.content
                response_tokens = count_tokens(response_data, "cl100k_base")
                return response_data, response_tokens
                
        return None, 0
