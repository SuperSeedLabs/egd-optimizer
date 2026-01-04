import os
from typing import Optional, Dict, Any, Type
import openai
from dotenv import load_dotenv
import torch.nn as nn
from agent.egd.egd_registry import register_module, ModuleRegistryDB

# NOTE: agent can generate and register modules with the registry_db
# NOTE: this is a quick test 


def create_module_from_openai(
    module_name: str,
    module_description: str,
    registry_db: Optional["ModuleRegistryDB"] = None
) -> Optional[Type[nn.Module]]:
    """
    Create a PyTorch module using OpenAI's O1 model based on a description.
    
    Args:
        module_name: Name for the new module
        module_description: Detailed description of the module to create
        registry_db: Optional ModuleRegistryDB instance to register the module
        
    Returns:
        Optional[Type[nn.Module]]: The created module class if successful, None otherwise
    """
    # Load API key from environment
    load_dotenv()
    api_key = os.getenv("OPENAI")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return None
    
    # Configure OpenAI client
    openai.api_key = api_key
    
    # Construct the prompt
    prompt = f"""
    Create a PyTorch neural network module named {module_name} with the following description:
    
    {module_description}
    
    Please implement the module as a class that inherits from nn.Module. Include:
    1. Proper initialization with clear parameter documentation
    2. Forward method implementation
    3. Comprehensive docstrings
    4. Type hints
    5. Any helper methods needed
    6. Do NOT include imports in the code block as they will be added later.
    
    Return your response in the following JSON format with " instead of ':
    {{
        "module_code": "... full Python code for the module ...",
        "description": "... a concise description of what the module does ...",
        "categories": ["... relevant categories for this module ..."],
        "tags": ["... relevant tags for this module ...", "ai-generated"],
        "example_code": "... code snippet showing how to instantiate and use the module ...",
        "example_inputs": {{
            "... parameter name ...": "... example value ..."
            // Include typical values for the main parameters
        }}
    }}
    
    Here are a few examples of well-structured PyTorch modules:
    
    Example 1 - Simple MLP:
    ```python
    class CustomMLP(nn.Module):
        '''
        A simple MLP with configurable hidden dimensions and dropout.
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            hidden_dim (int, optional): Hidden dimension size. Defaults to 128.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        '''
        def __init__(self, **kwargs):
            super().__init__()
            
            if 'in_features' not in kwargs or 'out_features' not in kwargs:
                raise ValueError("Missing required parameters 'in_features' or 'out_features'")
            
            in_features = kwargs['in_features']
            out_features = kwargs['out_features']
            hidden_dim = kwargs.get('hidden_dim', 128)
            dropout = kwargs.get('dropout', 0.1)
            
            self.fc1 = nn.Linear(in_features, hidden_dim)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, out_features)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            '''
            Forward pass through the MLP.
            
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_features)
                
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, out_features)
            '''
            x = self.fc1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    ```
    
    Example 2 - Attention Mechanism:
    ```python
    class GroupedQueryAttention(nn.Module):
        '''
        Grouped-query attention mechanism as used in Llama2.
        
        Args:
            dim (int): Feature dimension
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            num_kv_heads (int, optional): Number of key/value heads for grouped-query attention. Defaults to 4.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        '''
        def __init__(self, **kwargs):
            super().__init__()
            
            if 'dim' not in kwargs:
                raise ValueError("Missing required parameter 'dim'")
                
            dim = kwargs['dim']
            num_heads = kwargs.get('num_heads', 8)
            num_kv_heads = kwargs.get('num_kv_heads', 4)
            dropout = kwargs.get('dropout', 0.1)
            
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = dim // num_heads

            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, self.head_dim * num_kv_heads)
            self.v_proj = nn.Linear(dim, self.head_dim * num_kv_heads)
            self.o_proj = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            '''
            Forward pass through the attention mechanism.
            
            Args:
                x (torch.Tensor): Input tensor
                
            Returns:
                torch.Tensor: Output tensor
            '''
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            # Reshape for attention computation
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)

            # Compute attention with grouped queries
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            out = attn @ v
            return self.o_proj(out.reshape(*x.shape))
    ```
    """
    
    try:
        print("Calling OpenAI API...")
        # Call OpenAI API
        response = openai.chat.completions.create(
            model="o1",
            messages=[
                {"role": "system", "content": "You are an expert PyTorch developer who writes clean, efficient neural network modules."},
                {"role": "user", "content": prompt}
            ],
            # temperature=0.2,
            response_format={"type": "json_object"},
            # max_tokens=2000
        )
        
        print("OpenAI API call successful. Processing response...")
        # Extract the JSON response
        import json
        response_content = response.choices[0].message.content.strip()
        response_data = json.loads(response_content)

        print(f"Response data: {response_data}")
        
        # Extract the module code and metadata
        module_code = response_data["module_code"]
        module_description = response_data.get("description", module_description)
        module_categories = response_data.get("categories", ["ai_generated"])
        module_tags = response_data.get("tags", ["openai", "o1", "ai-generated"])
        module_example_code = response_data.get("example_code", f"model = {module_name}()")
        module_example_inputs = response_data.get("example_inputs", {})
        
        print("Executing generated module code...")
        # Create a namespace and execute the code
        namespace = {}
        # Store the module code without executing it
        module_class = None
        
        # Create a placeholder class that inherits from nn.Module
        class ModulePlaceholder(nn.Module):
            def __init__(self):
                super().__init__()
                
            def forward(self, x):
                raise NotImplementedError("This is a placeholder module that hasn't been executed")
        
        # Use the placeholder as our module class
        module_class = type(module_name, (ModulePlaceholder,), {})
        
        print(f"Created placeholder for module: {module_name}")
        
        # Register the module if registry_db is provided
        if registry_db and registry_db.connected:
            print("Registering module in the database...")
            module_id = registry_db.save_module(
                module_class=module_class,
                name=module_name,
                description=module_description,
                category=module_categories[0] if module_categories else "ai_generated",
                tags=module_tags,
                example_code=module_example_code,
                example_inputs=module_example_inputs,
                source_code=module_code  # Store the source code for later execution
            )
            
            if module_id:
                print(f"Module registered in database with ID: {module_id}")
                
                # Also register in the global MODULE_REGISTRY
                register_module(module_name)(module_class)
                registry_db.register_module_to_registry(module_id)
                print(f"Module registered in global MODULE_REGISTRY as '{module_name}'")
        
        return module_class
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


if __name__ == "__main__":
    # Test the OpenAI module creation
    registry = ModuleRegistryDB()

    if registry.connected:
        print("\n=== Testing Module Registry Database ===\n")
        
        # Example usage
        module_description = """
        A novel never seen before module for BabyLM classification. It uses a novel never seen before architecture to
        utterly destroy the competition on BabyLM.
        """
        
        vit_module = create_module_from_openai(
            module_name="2NovelModuleBabyLM",
            module_description=module_description,
            registry_db=registry
        )
        
        if vit_module:
            print("Module creation successful!")
            
            # Test the module if possible
            try:
                import torch
                model = vit_module()
                test_input = torch.randn(1, 3, 224, 224)  # Single RGB image
                output = model(test_input)
                print(f"Test successful! Output shape: {output.shape}")
            except Exception as e:
                print(f"Error testing module: {e}")
        
        # Close the registry connection
        if registry.connected:
            registry.close()
    else:
        print("Could not connect to MongoDB. Skipping database tests.")


