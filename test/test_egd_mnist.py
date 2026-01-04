# imports
import asyncio
import random
import time
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from agent.egd.egd import EGD
from agent.egd.egd_registry import register_module, ModuleConfig, ModuleRegistryDB
from agent.egd.utils import print_elapsed_time


# TODOs:
# 1. Integrate everything together and more with the agent framework
# 2. do real benchmarking on Agent EGD for LLM fine-tuning https://github.com/snap-stanford/MLAgentBench/tree/main BabyLM
# 3. add a way to increase or decrease the delta in hyperparams changes. a sort of acceleration factor to explore faster.

# QUESTION:
# 1. What is the right temperature for module creation?

# Define the MnistCNN class OUTSIDE of the main function for better serialization
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
        norm_type = kwargs.get('norm_type', None)

        # Handle activation - convert string to module if needed
        activation = kwargs.get('activation', nn.ReLU())
        if isinstance(activation, str):
            if activation == 'ReLU':
                activation = nn.ReLU()
            elif activation == 'LeakyReLU':
                activation = nn.LeakyReLU()
            elif activation == 'GELU':
                activation = nn.GELU()
            elif activation == 'Sigmoid':
                activation = nn.Sigmoid()
            elif activation == 'Tanh':
                activation = nn.Tanh()
            else:
                activation = nn.ReLU()

        # Handle output activation - convert string to module if needed
        output_activation = kwargs.get('output_activation', None)
        if isinstance(output_activation, str):
            if output_activation == 'Softmax':
                output_activation = nn.Softmax(dim=-1)
            elif output_activation == 'LogSoftmax':
                output_activation = nn.LogSoftmax(dim=-1)
            elif output_activation == 'Sigmoid':
                output_activation = nn.Sigmoid()
            elif output_activation == 'Tanh':
                output_activation = nn.Tanh()

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
            layers.append(nn.LayerNorm([conv_channels[1], self.input_shape[1] // 2, self.input_shape[2] // 2]))
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
        if output_activation is not None:
            layers.append(output_activation)

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


def load_mnist(
        data_percent: float = 1.0,
        device: torch.device = None,
        seed: Optional[int] = None
) -> Tuple[List, List, List, Tuple[int, int, int], Tuple[int]]:
    """Load and preprocess the MNIST dataset.
    
    Args:
        data_percent: Fraction of dataset to use (default: 1.0)
        device: PyTorch device to use (default: None)
        seed: Random seed for reproducibility (default: None)
        
    Returns:
        Tuple containing:
            - training_data: List of (input, target) tensor tuples for training
            - validation_data: List of (input, target) tensor tuples for validation
            - test_data: List of (input, target) tensor tuples for testing
            - input_shape: Shape of input images (28, 28, 1)
            - output_shape: Shape of output classes (10,)
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading MNIST data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_train = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform
    )

    train_size = int(len(mnist_train) * data_percent)
    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=train_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None
    )

    images, labels = next(iter(train_loader))
    images = images.view(train_size, -1).to(device)
    labels_onehot = torch.zeros(train_size, 10, device=device)
    labels_onehot.scatter_(1, labels.unsqueeze(1).to(device), 1)

    training_data = list(zip(images, labels_onehot))

    split_idx = int(len(training_data) * 0.2)
    validation_data = training_data[:split_idx]
    training_data = training_data[split_idx:]

    test_size = int(len(mnist_test) * data_percent)
    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=test_size, shuffle=False,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None
    )

    images, labels = next(iter(test_loader))
    images = images.view(test_size, -1).to(device)
    labels_onehot = torch.zeros(test_size, 10, device=device)
    labels_onehot.scatter_(1, labels.unsqueeze(1).to(device), 1)

    test_data = list(zip(images, labels_onehot))

    print(
        f'Loaded {len(training_data)} training examples, {len(validation_data)} validation examples, and {len(test_data)} testing examples.')

    return training_data, validation_data, test_data, (28, 28, 1), (10,)


def main():
    '''main of the program'''

    # Define a unique module name for database test
    db_module_name = "mnist_cnn_db_test"

    # Initialize database connection
    registry = ModuleRegistryDB()

    if registry.connected:
        print(f"\n=== Registering '{db_module_name}' module to database ===\n")

        # First, check if module exists and delete it to start fresh
        existing = registry.modules_collection.find_one({"name": db_module_name})
        if existing:
            print(f"Found existing module with name '{db_module_name}', deleting it first")
            registry.delete_module(str(existing["_id"]))

        # Save the module with serializable parameters
        module_id = registry.save_module(
            module_class=MnistCNN,
            name=db_module_name,
            description="CNN module specifically designed for MNIST dataset (DB version)",
            category="cnn",
            tags=["mnist", "cnn", "classification", "test"],
            example_code=f"model = {db_module_name}(in_features=(1, 28, 28), out_features=10)",
            example_inputs={
                "in_features": (1, 28, 28),
                "out_features": 10,
                "conv_channels": [8, 16],  # Smaller model (reduced from [32, 64])
                "fc_features": [32],  # Smaller model (reduced from [128])
                "dropout": 0.2,
                "norm_type": "batchnorm",
                "activation": "ReLU",
                "output_activation": "Softmax"
            }
        )

        if module_id:
            print(f"\nSaved '{db_module_name}' module to database with ID: {module_id}")

            # Create a much smaller network configuration to reduce memory usage
            db_cnn_config = [
                ModuleConfig(
                    module_type=db_module_name,
                    kwargs={
                        'in_features': (1, 28, 28),
                        'out_features': 10,
                        'conv_channels': [8, 16],  # Smaller model (reduced from [32, 64])
                        'fc_features': [32],  # Smaller model (reduced from [128])
                        'dropout': 0.2,
                        'norm_type': 'batchnorm',
                        'activation': 'ReLU',
                        'output_activation': 'Softmax',
                    }
                )
            ]

            # Close connection before EGD initialization
            registry.close()

            # Use much smaller data size and fewer epochs for testing
            meta_config = {
                'POPULATION_SIZE': 1,  # Minimum population
                'GENERATIONS': 1,  # Just one generation for testing
                'EPOCHS': 2,  # Minimum epochs for testing
                'DEBUG': True,
                'LAYERS_CONFIG_INIT': db_cnn_config,
                'LAYERS_CONFIG_OPTIONS': [db_cnn_config],
                'BATCH_SIZE_INIT': 16,  # Use very small batches (added to reduce memory)
            }

            # Load tiny fraction of MNIST data
            (training_data, validation_data, test_data, _, _) = load_mnist(data_percent=0.01)  # Use only 1% of data

            try:
                # Initialize EGD with our config
                egd = EGD(
                    training_data=training_data,
                    validation_data=validation_data,
                    loss_fn=nn.CrossEntropyLoss(),
                    meta_config=meta_config,
                    seed=42
                )

                print('\nRunning the population based training with database module\n')

                # Train the model
                best_net, most_acc = asyncio.run(egd.train())

                # Test results with smaller test set
                print('\nTesting the model loaded from database...\n')
                # Use only the first 100 test samples
                small_test_data = test_data[:100]
                accuracy = 100 * egd.test_net(best_net, small_test_data, acc_report=True)[1]
                print(f'\nAccuracy with database module: {accuracy:.2f}%\n')

                print("Database module test completed successfully!")

            except Exception as e:
                print(f"Error during EGD with database module: {e}")
                import traceback
                traceback.print_exc()

            # Clean up the test module
            print("\n=== Cleaning up database ===")
            registry = ModuleRegistryDB()
            if registry.connected:
                if registry.delete_module(module_id):
                    print(f"Successfully deleted test module from database")
                registry.close()
        else:
            print("Failed to save module to database. Skipping database test.")
            registry.close()
    else:
        print("Could not connect to MongoDB. Running the standard test only.")

    # Run the standard test with local modules and reduced memory usage
    print("\n=== Running standard test with local modules ===\n")

    # Use smaller networks for the standard test
    cnn_config = [
        ModuleConfig(
            module_type='mnist_cnn',
            kwargs={
                'in_features': (1, 28, 28),
                'out_features': 10,
                'conv_channels': [8, 16],  # Reduced from [32, 64]
                'fc_features': [32],  # Reduced from [128]
                'dropout': 0.2,
                'activation': nn.ReLU(),
                'norm_type': 'batchnorm',
                'output_activation': nn.Softmax(dim=-1),
            }
        )
    ]

    # Smaller variants
    cnn_config_small = [
        ModuleConfig(
            module_type='mnist_cnn',
            kwargs={
                'in_features': (1, 28, 28),
                'out_features': 10,
                'conv_channels': [4, 8],  # Very small model
                'fc_features': [16],
                'dropout': 0.1,
                'activation': nn.ReLU(),
                'norm_type': 'batchnorm',
                'output_activation': nn.Softmax(dim=-1),
            }
        )
    ]

    # Configuration dictionary for local test with reduced resources
    meta_config = {
        'POPULATION_SIZE': 2,  # Reduced from 3
        'GENERATIONS': 2,  # Reduced from 6
        'EPOCHS': 5,  # Reduced from 100
        'DEBUG': True,
        'LAYERS_CONFIG_INIT': cnn_config,
        'LAYERS_CONFIG_OPTIONS': [
            cnn_config_small,
        ],
        'BATCH_SIZE_INIT': 16,  # Use very small batches
    }

    # Load MNIST data and initialize EGD with smaller dataset
    (training_data, validation_data, test_data, _, _) = load_mnist(data_percent=0.05)  # 5% of data

    egd = EGD(
        training_data=training_data,
        validation_data=validation_data,
        loss_fn=nn.CrossEntropyLoss(),
        meta_config=meta_config,
        seed=42
    )

    # set log path
    egd.log_path = f'logs/mnist.csv'

    print('\nRunning the population based training\n')

    # Start timing
    start_time = time.time()

    # asyncio.run() executes the coroutine egd.train() in an event loop,
    # allowing concurrent training of multiple networks in the population
    best_net, most_acc = asyncio.run(egd.train())

    # End timing
    end_time = time.time()
    training_time = end_time - start_time

    print_elapsed_time(training_time)

    print(f'Configuration: Population size={meta_config["POPULATION_SIZE"]}, '
          f'Generations={meta_config["GENERATIONS"]}, '
          f'Epochs={meta_config["EPOCHS"]}')

    # Test with smaller test set
    small_test_data = test_data[:100]  # Use only 100 test samples

    # test the best performing network
    print('\nTesting the best performing network...\n')
    accuracy = 100 * egd.test_net(best_net, small_test_data, acc_report=True)[1]
    print('\nTesting complete\n')
    print(f'\nAccuracy: {accuracy:.2f}%\n')
    print(f'Number of parameters: {best_net.n_params}\n')
    print(f'Network architecture: {best_net}\n')

    # test the most accurate network
    print('\nTesting the most accurate network...\n')
    accuracy = 100 * egd.test_net(most_acc, small_test_data, acc_report=True)[1]
    print('\nTesting complete\n')
    print(f'\nAccuracy: {accuracy:.2f}%\n')
    print(f'Number of parameters: {most_acc.n_params}\n')
    print(f'Network architecture: {most_acc}\n')


if __name__ == '__main__':
    main()
