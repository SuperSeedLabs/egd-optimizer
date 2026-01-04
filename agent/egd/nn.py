import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from agent.egd.egd_registry import register_module, ModuleConfig, MODULE_REGISTRY, ModuleRegistryDB


class NeuralNetwork(nn.Module):
    """Advanced Generalized Neural Network Class supporting various layer types and configurations.
    
    Supports:
      • Common layer types including linear, convolutional (1D, 2D), attention, transformer blocks, etc.
      • Loading and saving configuration to/from YAML files for reproducibility of architectures.
      • Popular model configurations (e.g., LLMs, object detectors, agents) can be constructed from YAML.
      • Dynamic registration of custom modules via decorator pattern

    Attributes:
        net_id (int): Unique identifier for this network.
        layers_config (List[ModuleConfig]): List of layer configurations.
        learning_rate (float): Learning rate for optimization.
        momentum (float): Momentum coefficient for optimization.
        weight_decay (float): Weight decay coefficient.
        debug (bool): Whether to print debug information.
        norm_type (Optional[str]): Type of normalization to apply after layers ('batchnorm' or 'layernorm').
        dropout_p (float): Dropout probability.
        model (nn.Sequential): Sequential container of layer objects.
        device (torch.device): Device to run computations on (CPU or GPU).
        optimizer (torch.optim.Optimizer): Optimizer instance for training.
        n_params (int): Number of trainable parameters.
    """

    def __init__(
            self,
            net_id: int,
            hyperparams: dict,
            debug: bool = True
    ) -> None:
        """Initialize the Neural Network from a hyperparams dictionary.
        
        Args:
            net_id (int): Unique identifier for this network.
            hyperparams (dict): Dictionary containing necessary parameters, e.g.:
                {
                    'LAYERS_CONFIG': [ModuleConfig(module_type='linear', kwargs={'in_features': 784, 'out_features': 128, ...}), ...],
                        # List of ModuleConfig objects defining the elements of the network architecture.
                        # Each ModuleConfig specifies a layer type (e.g., 'linear', 'conv2d', 'lstm', 'transformer_block')
                        # and its parameters in the kwargs dictionary. The layers are built in the order they appear in this list.
                        # Available module types are registered in MODULE_REGISTRY or can be added via @register_module decorator.
                        # The input and output shapes are inferred from the layers configuration.
                        # The activation and output activation are inferred from the layers configuration.
                        # The normalization type is inferred from the layers configuration.
                    'LEARNING_RATE': float,
                    'MOMENTUM': float,
                    'WEIGHT_DECAY': float,
                    'DROPOUT': float,
                    'NORM_TYPE': str,
                    'OPTIMIZER': torch optimizer class,
                    ...
                }
            debug (bool): Whether to print debug information.
        """
        super(NeuralNetwork, self).__init__()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if debug:
            print(f"Using device: {self.device}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Store network hyperparameters
        self.net_id = net_id
        self.layers_config = hyperparams.get('LAYERS_CONFIG', [])
        self.learning_rate = hyperparams['LEARNING_RATE']
        self.momentum = hyperparams['MOMENTUM']
        self.weight_decay = hyperparams['WEIGHT_DECAY']
        self.dropout_p = hyperparams.get('DROPOUT', 0.0)
        self.debug = debug

        # Create network layers
        self.model = self._build_model()
        self.to(self.device) # Move model to device

        # Build optimizer
        optimizer_class = hyperparams.get('OPTIMIZER', torch.optim.AdamW)
        optimizer_kwargs = {'lr': self.learning_rate, 'weight_decay': self.weight_decay}
        # Only add momentum for optimizers that support it (like SGD)
        if optimizer_class.__name__ == 'SGD':
            optimizer_kwargs['momentum'] = self.momentum
        self.optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)

        # Number of parameters
        self.n_params = self.num_params()

    def num_params(self) -> int:
        """Calculate the number of trainable parameters in the network.
        
        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _build_model(self) -> nn.Sequential:
        """Build network layers based on the stored layer configurations using the registry pattern.
        
        This method constructs the neural network architecture by iterating through the layer
        configurations provided in self.layers_config. For each configuration, it:
        
        1. Attempts to find the module in the MODULE_REGISTRY dictionary first, which contains
           custom registered modules (registered via the @register_module decorator)
        2. If not found locally, tries to load it from the MongoDB database registry
        3. If the module_type is 'custom', it extracts the layer_class from kwargs and uses it directly
        4. As a fallback, tries to find the module in torch.nn standard library
        5. Adds dropout layers after each module if dropout probability is greater than 0
        
        The registry pattern allows for extensibility by enabling custom modules to be registered
        and used alongside standard PyTorch modules.
        
        Raises:
            ValueError: If the specified module_type is not found in the registry, database,
                       not 'custom', and not available in torch.nn
        Returns:
            nn.Sequential: Sequential container of layer objects.
        """
        
        # Initialize database connection
        db_registry = ModuleRegistryDB()
        
        layers = []
        for i, config in enumerate(self.layers_config):
            module_type = config.module_type
            kwargs = config.kwargs or {}

            # Try to get layer class from registry
            if module_type in MODULE_REGISTRY:
                # Use custom module from our registry (registered with @register_module decorator)
                layer_class = MODULE_REGISTRY[module_type]
                layer = layer_class(**kwargs)
            elif module_type == 'custom':
                # Handle custom layers that provide their own class
                # Extract and remove layer_class from kwargs to prevent passing it to the constructor
                layer_class = kwargs.pop('layer_class')
                layer = layer_class(**kwargs)
            else:
                # Try to load from database if connected
                db_module = None
                if db_registry.connected:
                    if self.debug:
                        print(f"Module '{module_type}' not found locally, attempting to load from database...")
                    db_module = db_registry.load_module_by_name(module_type)
                    
                if db_module is not None:
                    # Register the module locally for future use
                    MODULE_REGISTRY[module_type] = db_module
                    layer = db_module(**kwargs)
                    if self.debug:
                        print(f"Successfully loaded module '{module_type}' from database")
                else:
                    # Try to get layer directly from torch.nn as fallback
                    try:
                        # Dynamically access standard PyTorch layers like nn.Linear, nn.Conv2d, etc.
                        layer_class = getattr(nn, module_type)
                        layer = layer_class(**kwargs)
                    except (AttributeError, TypeError):
                        raise ValueError(f"Unsupported module type: {module_type}. "
                                         f"Not found in local registry, database, or torch.nn")

            layers.append(layer)

            # Insert dropout between layers (skip after the final layer)
            if self.dropout_p > 0 and i < len(self.layers_config) - 1:
                layers.append(nn.Dropout(self.dropout_p))
        
        # Close database connection
        if db_registry.connected:
            db_registry.close()
        
        return nn.Sequential(*layers)

    def print_network(self) -> None:
        """Print information about the network architecture and parameters.
        
        Displays:
        1. The layer configurations 
        2. The total number of trainable parameters
        
        Returns:
            None
        """
        print(f"Network ID: {self.net_id}")
        print(f"Total parameters: {self.n_params}")
        print(f"Layer configurations:")
        for i, config in enumerate(self.layers_config):
            print(f"  Layer {i}: {config.module_type} - {config.kwargs}")


    def save(self, filename: Union[str, None] = None) -> None:
        """Save the model's state dictionary to a file.
        
        Saves the model's parameters (weights and biases) to a file using
        PyTorch's save functionality.

        Args:
            filename: Optional path to save the model. If not provided,
                     defaults to 'model_{net_id}.pt'

        Returns:
            None
        """
        # Use default filename if none provided
        filename = filename or f'model_{self.net_id}.pt'

        # Save model state dictionary using PyTorch's save
        torch.save(self.model.state_dict(), filename)
        if self.debug: print(f"Model saved to {filename}")

    def load(self, filename: str) -> None:
        """Load a saved model's state dictionary from a file.
        
        Loads previously saved model parameters from a file using PyTorch's load functionality.

        Args:
            filename: Path to the saved model file (.pt extension)

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified file does not exist
            RuntimeError: If the loaded state dict is not compatible with current model
        """
        # Load the saved state dictionary using PyTorch's load
        state_dict = torch.load(filename, map_location=self.device)

        # Load the state dictionary into the model
        self.model.load_state_dict(state_dict)

        # Set model to evaluation mode
        self.eval()
        if self.debug: print(f"Model loaded from {filename}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network.
        
        Sequentially passes the input through all layers in the model.

        Args:
            x: Input tensor matching the expected input shape

        Returns:
            torch.Tensor: Output tensor after passing through all network layers
        """
        # Move input to device
        x = x.to(self.device)
        return self.model(x)

    def predict(self, instance: torch.Tensor) -> torch.Tensor:
        """Make predictions using the neural network model.
        
        Performs forward pass through the network in evaluation mode.

        Args:
            instance: Input tensor matching the expected input shape

        Returns:
            torch.Tensor: Output tensor containing model predictions
        """
        # Set model to evaluation mode
        self.eval()

        # Make predictions using forward pass without gradient tracking
        with torch.no_grad():
            predictions = self.forward(instance)

        return predictions

    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.9):
        """Delegates to the generate method of the first module in the model that has one."""
        for module in self.model:
            if hasattr(module, 'generate'):
                return module.generate(idx, max_new_tokens, temperature, top_p)
        raise AttributeError("No module with generate method found in the model")


if __name__ == '__main__':

    print("Quick test of MNIST CNN created module")
    # Custom CNN module for MNIST
    @register_module('mnist_cnn')
    class MnistCNN(nn.Module):
        """Custom CNN module specifically designed for MNIST dataset.
        
        This module automatically builds a convolutional neural network
        based on the input shape and output shape provided. It's optimized
        for the MNIST dataset with 28x28 grayscale images.
        
        Attributes:
            input_shape (Tuple): Shape of input images (channels, height, width)
            output_shape (int): Number of output classes
            conv_channels (List[int]): Number of channels in each conv layer
            fc_features (List[int]): Number of features in each fully connected layer
            dropout (float): Dropout probability
            activation (nn.Module): Activation function to use
            norm_type (str): Type of normalization to use
            output_activation (nn.Module): Output activation function to use
        """

        def __init__(self, **kwargs):
            super().__init__()

            # Get required parameters
            if 'in_features' not in kwargs:
                raise ValueError("Missing required parameter 'in_features'")
            if 'out_features' not in kwargs:
                raise ValueError("Missing required parameter 'out_features'")

            # Store shapes
            if isinstance(kwargs['in_features'], (tuple, list)):
                self.input_shape = kwargs['in_features']
            else:
                raise ValueError("Input features must be a tuple (channels, height, width)")

            if isinstance(kwargs['out_features'], (tuple, list)):
                self.output_shape = kwargs['out_features'][0]
            else:
                self.output_shape = kwargs['out_features']

            # Get parameters from kwargs with defaults
            conv_channels = kwargs.get('conv_channels', [32, 64])
            fc_features = kwargs.get('fc_features', [128])
            dropout = kwargs.get('dropout', 0.5)
            activation = kwargs.get('activation', nn.ReLU())
            norm_type = kwargs.get('norm_type', None)

            # Build the CNN architecture
            layers = []

            # First convolutional layer
            layers.append(nn.Conv2d(self.input_shape[0], conv_channels[0], kernel_size=3, padding=1))
            # Add normalization if specified
            if norm_type == 'batchnorm':
                layers.append(nn.BatchNorm2d(conv_channels[0]))
            elif norm_type == 'layernorm':
                layers.append(nn.LayerNorm([conv_channels[0], self.input_shape[1], self.input_shape[2]]))
            # Add activation
            layers.append(activation)
            layers.append(nn.MaxPool2d(2))

            # Second convolutional layer
            layers.append(nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1))
            # Add normalization if specified
            if norm_type == 'batchnorm':
                layers.append(nn.BatchNorm2d(conv_channels[1]))
            elif norm_type == 'layernorm':
                layers.append(nn.LayerNorm([conv_channels[1], self.input_shape[1]//2, self.input_shape[2]//2]))
            # Add activation
            layers.append(activation)
            layers.append(nn.MaxPool2d(2))

            # Calculate the size after convolutions and pooling
            conv_output_size = (self.input_shape[1] // 4) * (self.input_shape[2] // 4) * conv_channels[1]

            # Flatten layer
            layers.append(nn.Flatten())

            # Fully connected layers
            current_size = conv_output_size
            for fc_size in fc_features:
                layers.append(nn.Linear(current_size, fc_size))
                # Add normalization if specified
                if norm_type == 'batchnorm':
                    layers.append(nn.BatchNorm1d(fc_size))
                elif norm_type == 'layernorm':
                    layers.append(nn.LayerNorm(fc_size))
                # Add activation
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
                current_size = fc_size

            # Output layer
            layers.append(nn.Linear(current_size, self.output_shape))
            # Add output activation if specified
            if 'output_activation' in kwargs and kwargs['output_activation'] is not None:
                layers.append(kwargs['output_activation'])

            # Create sequential model
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            # Ensure input has the right shape (batch_size, channels, height, width)
            if len(x.shape) == 2:
                # If input is flattened, reshape it
                batch_size = x.shape[0]
                x = x.view(batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            elif len(x.shape) == 3:
                # If input is (batch_size, height, width), add channel dimension
                x = x.unsqueeze(1)

            return self.model(x)
        
    
    # Test parameters with MNIST CNN
    mnist_test_hyperparams = {
        'LAYERS_CONFIG': [
            ModuleConfig(
                module_type='mnist_cnn',
                kwargs={
                    'in_features': (1, 28, 28),
                    'out_features': 10,
                    'conv_channels': [32, 64],
                    'fc_features': [128],
                    'dropout': 0.5,
                    'activation': nn.ReLU(),
                    'norm_type': 'batchnorm',
                    'output_activation': nn.Softmax(dim=-1),
                }
            )
        ],
        'LEARNING_RATE': 0.001,
        'MOMENTUM': 0.9,
        'WEIGHT_DECAY': 0.0001,
        'DROPOUT': 0.1,
        'OPTIMIZER': torch.optim.Adam,
        'BATCH_SIZE': 128,
    }

    # Create test network with MNIST CNN
    mnist_net = NeuralNetwork(net_id=0, hyperparams=mnist_test_hyperparams, debug=True)

    print(f"\nMNIST CNN network created successfully")
    print(f"Number of parameters: {mnist_net.n_params}")
    print(f"Network architecture:\n{mnist_net}")