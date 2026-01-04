import base64
import importlib
import inspect
import json
import os
import pickle
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Union, Tuple, List, Optional, Dict, Type, Any

import torch
import torch.nn as nn
from bson.objectid import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

# Global registry for layer types
MODULE_REGISTRY: Dict[str, Type[nn.Module]] = {
    # linear modules
    'linear': nn.Linear,
    # convolutional modules
    'conv1d': nn.Conv1d,
    'conv2d': nn.Conv2d,
    'conv3d': nn.Conv3d,
    # embedding modules
    'embedding': nn.Embedding,
    # recurrent modules
    'lstm': nn.LSTM,
    'gru': nn.GRU,
    # attention modules
    'multihead_attention': nn.MultiheadAttention,
    # transformer modules
    'transformer_encoder': nn.TransformerEncoder,
    'transformer_decoder': nn.TransformerDecoder,
    'transformer_encoder_layer': nn.TransformerEncoderLayer,
    'transformer_decoder_layer': nn.TransformerDecoderLayer,
    'transformer': nn.Transformer,
    # normalization modules
    'layernorm': nn.LayerNorm,
    'batchnorm': nn.BatchNorm1d,
    # dropout modules
    'dropout': nn.Dropout,
    # activation modules
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'softmax': nn.Softmax,
}


@dataclass
class ModuleConfig:
    """
    Configuration for a neural network module.
    
    Attributes:
        module_type (str): Type of module ('linear', 'conv1d', 'conv2d', 'attention', 'embedding',
            'transformer', 'transformer_block', 'lstm', 'gru', 'custom', etc.)
        kwargs (Dict[str, Any]): Module-specific parameters including input and output dimensions.
            Each module has their own args so it is best to not force a specific format.
        
    """
    module_type: str
    kwargs: Dict[str, Any] = None


def register_module(name: str) -> Callable:
    """Decorator to register a module type in the global registry.
    
    Args:
        name (str): Name to register the module under
        
    Returns:
        Callable: Decorator function that registers the module
    """

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODULE_REGISTRY[name] = cls
        return cls

    return decorator


def get_module(name: str) -> Optional[Type[nn.Module]]:
    """Get a module from the registry by name.
    
    Args:
        name (str): Name of the registered module
        
    Returns:
        Optional[Type[nn.Module]]: The module class if found, None otherwise
    """
    return MODULE_REGISTRY.get(name)


def list_modules() -> List[str]:
    """Get list of all registered module names.
    
    Returns:
        List[str]: List of registered module names
    """
    return list(MODULE_REGISTRY.keys())


def delete_module(name: str) -> bool:
    """Delete a module from the registry.
    
    Args:
        name (str): Name of module to delete
        
    Returns:
        bool: True if module was deleted, False if not found
    """
    if name in MODULE_REGISTRY:
        del MODULE_REGISTRY[name]
        return True
    return False


def save_registry(filepath: str) -> None:
    """Save the current registry to a JSON file.
    
    Args:
        filepath (str): Path to save the registry JSON file
        
    Note:
        Only saves the module names and their source (torch.nn or custom).
        Custom modules need to be re-registered after loading.
    """
    registry_data = {}
    for name, module in MODULE_REGISTRY.items():
        # Store module source (torch.nn or custom)
        if module.__module__.startswith('torch.nn'):
            registry_data[name] = {
                'source': 'torch.nn',
                'class_name': module.__name__
            }
        else:
            registry_data[name] = {
                'source': 'custom',
                'class_name': module.__name__
            }

    with open(filepath, 'w') as f:
        json.dump(registry_data, f, indent=2)


def load_registry(filepath: str) -> None:
    """
    Load registry configuration from a JSON file.
    
    Args:
        filepath (str): Path to the registry JSON file
        
    Note:
        Only loads torch.nn modules automatically.
        Custom modules need to be re-registered separately.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Registry file not found: {filepath}")

    with open(filepath, 'r') as f:
        registry_data = json.load(f)

    # Clear current registry
    MODULE_REGISTRY.clear()

    # Load modules
    for name, info in registry_data.items():
        if info['source'] == 'torch.nn':
            try:
                module = getattr(nn, info['class_name'])
                MODULE_REGISTRY[name] = module
            except AttributeError:
                print(f"Warning: Could not load torch.nn module {info['class_name']}")
        else:
            print(f"Note: Custom module {name} needs to be re-registered")


# registry of modules on cloud database
# NOTE: this is a temporary registry for testing purposes. 
# I am not convinced a database is the best way to do this. On one hand a database is
# good for storing metadata about modules and key module information. However, it doesn't give 
# a good way to load modules directly from the cloud and run them without passing through an 
# intermediate middle layer like String -> Module conversion. The issue with that middle layer 
# is that the environment needs to be the same for the string conversion to reproduce the module.
# A better way would be able to load the module directly from the cloud database without converting.
# Something maybe nn.save and nn.load can do with some files storage db, but right now it doesn't support 
# saving and loading modules like it does for models.


class ModuleRegistryDB:
    """
    A class for managing a registry of neural network modules in a MongoDB database.
    
    This class provides functionality to:
    - Store modules with their source code and metadata
    - Retrieve modules from the database
    - Search modules by various criteria
    - Test modules with sample inputs
    - Import modules directly into the current runtime
    
    Attributes:
        client (MongoClient): MongoDB client connection
        db (Database): MongoDB database instance
        modules_collection (Collection): Collection for storing module data
        connected (bool): Connection status to the database
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the ModuleRegistryDB with a MongoDB connection.
        
        Args:
            connection_string: MongoDB connection URI string. If None, will try to load from
                              MONGODB_URI in Coder/.env file.
        """
        # Try to load connection string from .env if not provided
        if connection_string is None:
            load_dotenv(os.path.join(os.path.dirname(__file__), '../../../.env'))
            connection_string = os.getenv('MONGODB_URI')

        if not connection_string:
            self.connected = False
            print("Warning: No MongoDB connection string provided. Running in offline mode.")
            return

        try:
            # Connect to MongoDB
            self.client = MongoClient(connection_string)
            self.db = self.client.module_registry
            self.modules_collection = self.db.modules

            # Create indexes for efficient querying
            self.modules_collection.create_index("name")
            self.modules_collection.create_index("tags")
            self.modules_collection.create_index("category")

            # Test connection
            self.client.admin.command('ping')
            self.connected = True
            print("Successfully connected to MongoDB module registry")
        except Exception as e:
            self.connected = False
            print(f"Failed to connect to MongoDB: {e}")

    def save_module(
            self,
            module_class: Type[nn.Module],
            name: str,
            description: str,
            category: str,
            tags: List[str],
            example_code: Optional[str] = None,
            example_inputs: Optional[Dict[str, Any]] = None,
            source_code: Optional[str] = None
    ) -> Optional[str]:
        """
        Save a PyTorch module to the database with metadata.
        
        Args:
            module_class: The PyTorch module class to save
            name: Unique name for the module
            description: Detailed description of the module
            category: Category for organization (e.g., 'attention', 'normalization')
            tags: List of searchable tags
            example_code: Example code showing how to use the module
            example_inputs: Dictionary of example inputs for testing the module
            source_code: Optional source code string. If None, will attempt to extract from module_class
            
        Returns:
            Optional[str]: The ObjectId of the inserted document as a string, or None if failed
        """
        if not self.connected:
            print("Error: Not connected to MongoDB")
            return None

        # Check if module with this name already exists
        existing = self.modules_collection.find_one({"name": name})
        if existing:
            print(f"Warning: Module with name '{name}' already exists. Use update_module instead.")
            return None

        # Get module source code if not provided
        if source_code is None:
            try:
                source_code = inspect.getsource(module_class)
                print(f"Source code for {module_class.__name__}:")
                print(source_code)
            except (TypeError, OSError):
                print(f"Warning: Could not retrieve source code for {module_class.__name__}")
                source_code = "# Source code not available"

        # Create a simple instance for serialization testing if possible
        serialized_instance = None
        try:
            # Try to create a minimal instance for testing
            signature = inspect.signature(module_class.__init__)
            params = {}
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                if param.default is not param.empty:
                    params[param_name] = param.default
                elif param.annotation is int:
                    params[param_name] = 64  # Default size for testing
                elif param.annotation is float:
                    params[param_name] = 0.1  # Default value for testing
                else:
                    params[param_name] = None

            # Create test instance and serialize
            test_instance = module_class(**{k: v for k, v in params.items() if v is not None})
            buffer = pickle.dumps(test_instance)
            serialized_instance = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Warning: Could not create test instance: {e}")

        # Prepare document for MongoDB
        module_doc = {
            "name": name,
            "class_name": module_class.__name__,
            "description": description,
            "category": category,
            "tags": tags,
            "source_code": source_code,
            "module_path": module_class.__module__,
            "serialized_instance": serialized_instance,
            "example_code": example_code,
            "example_inputs": example_inputs,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "pytorch_version": torch.__version__
        }

        # Insert into database
        try:
            result = self.modules_collection.insert_one(module_doc)
            print(f"Module '{name}' saved to database with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error saving module to database: {e}")
            return None

    def load_module(self, module_id: Union[str, ObjectId]) -> Optional[Type[nn.Module]]:
        """
        Load a module from the database by its ID.
        
        Args:
            module_id: The ObjectId of the module document as string or ObjectId
            
        Returns:
            Optional[Type[nn.Module]]: The loaded module class, or None if not found
        """
        if not self.connected:
            print("Error: Not connected to MongoDB")
            return None

        # Convert string ID to ObjectId if needed
        if isinstance(module_id, str):
            module_id = ObjectId(module_id)

        # Find the module in the database
        module_doc = self.modules_collection.find_one({"_id": module_id})
        if not module_doc:
            print(f"Module with ID {module_id} not found")
            return None

        return self._load_module_from_doc(module_doc)

    def load_module_by_name(self, name: str) -> Optional[Type[nn.Module]]:
        """
        Load a module from the database by its name.
        
        Args:
            name: The name of the module
            
        Returns:
            Optional[Type[nn.Module]]: The loaded module class, or None if not found
        """
        if not self.connected:
            print("Error: Not connected to MongoDB")
            return None

        # Find the module in the database
        module_doc = self.modules_collection.find_one({"name": name})
        if not module_doc:
            print(f"Module with name '{name}' not found")
            return None

        return self._load_module_from_doc(module_doc)

    def _load_module_from_doc(self, module_doc: Dict) -> Optional[Type[nn.Module]]:
        """
        Internal method to load a module from a document.
        
        Args:
            module_doc: The MongoDB document containing module data
            
        Returns:
            Optional[Type[nn.Module]]: The loaded module class, or None if loading failed
        """
        # First try to import the module directly if it's in a known path
        try:
            module = importlib.import_module(module_doc["module_path"])
            module_class = getattr(module, module_doc["class_name"])
            print(f"Successfully imported {module_doc['name']} from {module_doc['module_path']}")
            return module_class
        except (ImportError, AttributeError) as e:
            print(f"Could not import module directly: {e}")

        # If direct import fails, try to execute the source code
        try:
            source_code = module_doc["source_code"]
            namespace = {}
            exec(source_code, namespace)
            module_class = namespace[module_doc["class_name"]]
            print(f"Successfully loaded {module_doc['name']} from source code")
            return module_class
        except Exception as e:
            print(f"Error loading module from source code: {e}")

        # If all else fails, try to deserialize the instance
        if module_doc.get("serialized_instance"):
            try:
                serialized_data = base64.b64decode(module_doc["serialized_instance"])
                instance = pickle.loads(serialized_data)
                print(f"Successfully loaded {module_doc['name']} from serialized instance")
                return instance.__class__
            except Exception as e:
                print(f"Error deserializing module instance: {e}")

        return None

    def search_modules(
            self,
            query: Optional[str] = None,
            category: Optional[str] = None,
            tags: Optional[List[str]] = None,
            limit: int = 10
    ) -> List[Dict]:
        """
        Search for modules in the database.
        
        Args:
            query: Text to search in name and description
            category: Filter by category
            tags: Filter by tags (modules must have all specified tags)
            limit: Maximum number of results to return
            
        Returns:
            List[Dict]: List of matching module documents (without source code and serialized data)
        """
        if not self.connected:
            print("Error: Not connected to MongoDB")
            return []

        # Build the search filter
        search_filter = {}

        if query:
            search_filter["$or"] = [
                {"name": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}}
            ]

        if category:
            search_filter["category"] = category

        if tags:
            search_filter["tags"] = {"$all": tags}

        # Execute the search
        cursor = self.modules_collection.find(
            search_filter,
            # Exclude large fields from results
            {"source_code": 0, "serialized_instance": 0}
        ).limit(limit)

        return list(cursor)

    def register_module_to_registry(self, module_id: Union[str, ObjectId]) -> bool:
        """
        Load a module from the database and register it in the global MODULE_REGISTRY.
        
        Args:
            module_id: The ObjectId of the module document as string or ObjectId
            
        Returns:
            bool: True if successful, False otherwise
        """
        module_class = self.load_module(module_id)
        if not module_class:
            return False

        # Find the module document to get the name
        if isinstance(module_id, str):
            module_id = ObjectId(module_id)

        module_doc = self.modules_collection.find_one({"_id": module_id})
        if not module_doc:
            print(f"Module with ID {module_id} not found")
            return False

        # Register the module
        MODULE_REGISTRY[module_doc["name"]] = module_class
        print(f"Registered {module_doc['name']} in MODULE_REGISTRY")
        return True

    def test_module(
            self,
            module_id: Union[str, ObjectId],
            input_shape: Optional[tuple] = None
    ) -> Optional[nn.Module]:
        """
        Load a module from the database and test it with sample inputs.
        
        Args:
            module_id: The ObjectId of the module document as string or ObjectId
            input_shape: Shape of the test input tensor (default: (1, 64))
            
        Returns:
            Optional[nn.Module]: The instantiated module if successful, None otherwise
        """
        module_class = self.load_module(module_id)
        if not module_class:
            return None

        # Find the module document to get example inputs
        if isinstance(module_id, str):
            module_id = ObjectId(module_id)

        module_doc = self.modules_collection.find_one({"_id": module_id})
        if not module_doc:
            return None

        # Try to instantiate the module
        try:
            # Use example inputs if available
            if module_doc.get("example_inputs"):
                module_instance = module_class(**module_doc["example_inputs"])
            else:
                # Try to create with minimal parameters
                signature = inspect.signature(module_class.__init__)
                params = {}
                for param_name, param in signature.parameters.items():
                    if param_name == 'self':
                        continue
                    if param.default is not param.empty:
                        params[param_name] = param.default
                    elif param.annotation is int:
                        params[param_name] = 64  # Default size for testing
                    elif param.annotation is float:
                        params[param_name] = 0.1  # Default value for testing

                module_instance = module_class(**{k: v for k, v in params.items() if v is not None})

            # Test with a sample input
            if input_shape is None:
                input_shape = (1, 64)  # Default shape

            test_input = torch.randn(input_shape)
            output = module_instance(test_input)

            print(f"Module test successful!")
            print(f"Input shape: {test_input.shape}")
            print(f"Output shape: {output.shape}")

            return module_instance
        except Exception as e:
            print(f"Error testing module: {e}")
            return None

    def delete_module(self, module_id: Union[str, ObjectId]) -> bool:
        """
        Delete a module from the database.
        
        Args:
            module_id: The ObjectId of the module document as string or ObjectId
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            print("Error: Not connected to MongoDB")
            return False

        # Convert string ID to ObjectId if needed
        if isinstance(module_id, str):
            module_id = ObjectId(module_id)

        # Delete the module
        result = self.modules_collection.delete_one({"_id": module_id})
        if result.deleted_count == 1:
            print(f"Module with ID {module_id} deleted successfully")
            return True
        else:
            print(f"Module with ID {module_id} not found")
            return False

    def close(self):
        """Close the MongoDB connection."""
        if hasattr(self, 'client') and self.connected:
            self.client.close()
            self.connected = False
            print("MongoDB connection closed")

if __name__ == "__main__":
    # Test the ModuleRegistryDB class
    # Define a simple custom module for testing
    @register_module('custom_mlp')
    class CustomMLP(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            if 'in_features' not in kwargs:
                raise ValueError("Missing required parameter 'in_features'")
            if 'out_features' not in kwargs:
                raise ValueError("Missing required parameter 'out_features'")
                
            in_features = kwargs['in_features']
            out_features = kwargs['out_features']
            hidden_dim = kwargs.get('hidden_dim', 128)
            dropout = kwargs.get('dropout', 0.1)
            
            self.fc1 = nn.Linear(in_features, hidden_dim)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, out_features)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # Define Llama2 related modules
    @register_module('llama2_block')
    class Llama2Block(nn.Module):
        """Custom Llama2 block implementing key architectural components.
        
        Key features:
        - RMSNorm for normalization
        - Rotary positional embeddings
        - SwiGLU activation in MLP
        - Grouped Query Attention
        
        Attributes:
            dim (int): Feature dimension
            num_heads (int): Number of attention heads
            num_kv_heads (int): Number of key/value heads for grouped-query attention
            mlp_ratio (float): Ratio for MLP hidden dimension
            dropout (float): Dropout probability
        """

        def __init__(self, **kwargs):
            super().__init__()
            
            if 'dim' not in kwargs:
                raise ValueError("Missing required parameter 'dim'")
                
            dim = kwargs['dim']
            num_heads = kwargs.get('num_heads', 8)
            num_kv_heads = kwargs.get('num_kv_heads', 4)  # Grouped-query attention
            mlp_ratio = kwargs.get('mlp_ratio', 2.67)  # Llama2 default
            dropout = kwargs.get('dropout', 0.1)

            # RMSNorm layers (Llama2 uses RMSNorm instead of LayerNorm)
            self.norm1 = RMSNorm(dim=dim)
            self.norm2 = RMSNorm(dim=dim)

            # Grouped-query attention
            self.attention = GroupedQueryAttention(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dropout=dropout
            )

            # MLP block with SwiGLU activation
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                SwiGLU(),  # Llama2's activation function
                nn.Linear(mlp_hidden_dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            # Attention block with RMSNorm
            normed = self.norm1(x)
            x = x + self.attention(normed)

            # MLP block with RMSNorm
            x = x + self.mlp(self.norm2(x))
            return x


    @register_module('rms_norm')
    class RMSNorm(nn.Module):
        """Root Mean Square normalization used in Llama2"""

        def __init__(self, **kwargs):
            super().__init__()
            if 'dim' not in kwargs:
                raise ValueError("Missing required parameter 'dim'")
                
            dim = kwargs['dim']
            eps = kwargs.get('eps', 1e-6)
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            # Calculate RMS
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * rms * self.weight


    @register_module('swiglu')
    class SwiGLU(nn.Module):
        """SwiGLU activation function used in Llama2"""

        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, x):
            x, gate = x.chunk(2, dim=-1)
            return x * torch.sigmoid(gate) * 2.0


    @register_module('grouped_query_attention')
    class GroupedQueryAttention(nn.Module):
        """Grouped-query attention mechanism used in Llama2"""

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

        def forward(self, x):
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
    
    # Create registry instance
    registry = ModuleRegistryDB()
    
    if registry.connected:
        print("\n=== Testing Module Registry Database ===\n")
        
        # 1. Save the custom module to the database
        module_id = registry.save_module(
            module_class=CustomMLP,
            name="custom_mlp_test",
            description="A simple MLP with configurable hidden dimensions and dropout",
            category="feedforward",
            tags=["mlp", "feedforward", "test"],
            example_code="model = CustomMLP(in_features=784, out_features=10, hidden_dim=256)",
            example_inputs={
                "in_features": 784,
                "out_features": 10,
                "hidden_dim": 256,
                "dropout": 0.2
            }
        )
        
        print(f"\nSaved module with ID: {module_id}")
        
        # Save Llama2 related modules to the database
        llama2_block_id = registry.save_module(
            module_class=Llama2Block,
            name="llama2_block",
            description="Custom Llama2 block with RMSNorm, SwiGLU, and Grouped Query Attention",
            category="transformer",
            tags=["llama2", "transformer", "attention"],
            example_code="model = Llama2Block(dim=512, num_heads=8, num_kv_heads=4)",
            example_inputs={
                "dim": 512,
                "num_heads": 8,
                "num_kv_heads": 4,
                "mlp_ratio": 2.67,
                "dropout": 0.1
            }
        )
        print(f"\nSaved Llama2Block module with ID: {llama2_block_id}")
        
        rms_norm_id = registry.save_module(
            module_class=RMSNorm,
            name="rms_norm",
            description="Root Mean Square normalization used in Llama2",
            category="normalization",
            tags=["llama2", "normalization"],
            example_code="norm = RMSNorm(dim=512)",
            example_inputs={
                "dim": 512,
                "eps": 1e-6
            }
        )
        print(f"\nSaved RMSNorm module with ID: {rms_norm_id}")
        
        swiglu_id = registry.save_module(
            module_class=SwiGLU,
            name="swiglu",
            description="SwiGLU activation function used in Llama2",
            category="activation",
            tags=["llama2", "activation"],
            example_code="activation = SwiGLU()",
            example_inputs={}
        )
        print(f"\nSaved SwiGLU module with ID: {swiglu_id}")
        
        gqa_id = registry.save_module(
            module_class=GroupedQueryAttention,
            name="grouped_query_attention",
            description="Grouped-query attention mechanism used in Llama2",
            category="attention",
            tags=["llama2", "attention"],
            example_code="attn = GroupedQueryAttention(dim=512, num_heads=8, num_kv_heads=4)",
            example_inputs={
                "dim": 512,
                "num_heads": 8,
                "num_kv_heads": 4,
                "dropout": 0.1
            }
        )
        print(f"\nSaved GroupedQueryAttention module with ID: {gqa_id}")
        
        # 2. Search for the module
        print("\n=== Searching for modules with tag 'mlp' ===")
        results = registry.search_modules(tags=["mlp"])
        for result in results:
            print(f"Found: {result['name']} - {result['description']}")
        
        # 3. Load the module by ID
        print(f"\n=== Loading module by ID {module_id} ===")
        loaded_module_class = registry.load_module(module_id)
        print(f"Loaded module class: {loaded_module_class.__name__}")
        
        # 4. Test the module
        print("\n=== Testing the loaded module ===")
        module_instance = registry.test_module(
            module_id=module_id,
            input_shape=(32, 784)  # Batch of 32 MNIST flattened images
        )
        
        # 5. Register to global registry
        print("\n=== Registering module to global registry ===")
        registry.register_module_to_registry(module_id)
        registry.register_module_to_registry(llama2_block_id)
        registry.register_module_to_registry(rms_norm_id)
        registry.register_module_to_registry(swiglu_id)
        registry.register_module_to_registry(gqa_id)
        print(f"Available modules: {list_modules()}")
        
        # 6. Create a test network using ModuleConfig
        print("\n=== Creating a test network with ModuleConfig ===")
        test_hyperparams = {
            'LAYER_CONFIGS': [
                ModuleConfig(
                    module_type='custom_mlp',
                    kwargs={
                        'in_features': 784,
                        'out_features': 128,
                        'hidden_dim': 256,
                        'dropout': 0.2
                    }
                ),
                ModuleConfig(
                    module_type='linear',
                    kwargs={
                        'in_features': 128,
                        'out_features': 10
                    }
                )
            ],
            'LEARNING_RATE': 0.001,
            'WEIGHT_DECAY': 0.0001,
            'DROPOUT': 0.1
        }
        
        print(f"Test hyperparams created with custom_mlp module")
        print(f"Layer configs: {len(test_hyperparams['LAYER_CONFIGS'])} layers")
        for i, config in enumerate(test_hyperparams['LAYER_CONFIGS']):
            print(f"  Layer {i}: {config.module_type}")
        
        # 7. Clean up - delete only the test MLP module, keep Llama2 modules
        print("\n=== Cleaning up - deleting test module ===")
        if registry.delete_module(module_id):
            print("Test module deleted successfully")
        
        # Close the connection
        registry.close()
    else:
        print("Could not connect to MongoDB. Skipping database tests.")
        
        # Test the local registry functionality instead
        print("\n=== Testing Local Module Registry ===\n")
        
        # Register the custom module
        print(f"Registered modules before: {list_modules()}")
        
        # The module is already registered via the decorator
        print(f"Registered modules after: {list_modules()}")
        
        # Get and use a module
        custom_mlp_class = get_module('custom_mlp')
        test_mlp = custom_mlp_class(in_features=100, out_features=10)
        
        # Test the module
        test_input = torch.randn(32, 100)  # Batch of 32, 100 features
        output = test_mlp(test_input)
        
        print(f"Test successful!")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")