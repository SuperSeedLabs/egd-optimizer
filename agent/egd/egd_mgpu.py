# Imports
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Tuple, Optional, Dict, Union, Any, Callable
import copy  # Added for deep copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import agent.egd.utils as utils
from agent.egd.nn import NeuralNetwork as NN
from agent.egd.egd_registry import ModuleConfig, MODULE_REGISTRY

import ray
from ray.util.actor_pool import ActorPool
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray.cloudpickle as pickle


# nets should have the same loss function and the same optimizer. otherwise, 
# the search of the landscape will not be comparable.


# Add this Ray actor class definition before the EGD class
@ray.remote
class TrainingWorker:
    """Ray actor for distributed training of neural networks.
    
    This worker runs on a specific GPU and handles training tasks for assigned networks.
    
    Attributes:
        device_id (int): GPU device ID this worker runs on
        device (torch.device): PyTorch device object
        loss_fn (torch.nn.Module): Loss function to use for training
        debug (bool): Whether to print debug information
    """
    
    def __init__(self, device_id: int, loss_fn: torch.nn.Module, debug: bool = False):
        """Initialize the training worker.
        
        Args:
            device_id: GPU device ID to use
            loss_fn: Loss function for training
            debug: Whether to print debug information
        """
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.loss_fn = loss_fn
        self.debug = debug
        
        if self.debug:
            print(f"Initialized Ray worker on GPU {device_id}")
        
        # Cache for loaded data to prevent redundant serialization
        self.data_cache = {}
    
    def train_net_one_gen(self, net: NN, training_data: List, epochs: int, hyperparams: dict) -> NN:
        """Train a neural network for one generation.
        
        Args:
            net: Network to train
            training_data: Training dataset
            epochs: Number of epochs to train for
            hyperparams: Network hyperparameters
            
        Returns:
            NN: Trained neural network
        """
        # Move network to this worker's device
        net.to(self.device)
        
        # Set optimizer to use the network's parameters
        optimizer_class = hyperparams.get('OPTIMIZER', torch.optim.AdamW)
        optimizer_kwargs = {
            'lr': hyperparams['LEARNING_RATE'], 
            'weight_decay': hyperparams['WEIGHT_DECAY']
        }
        if optimizer_class.__name__ == 'SGD':
            optimizer_kwargs['momentum'] = hyperparams['MOMENTUM']
        net.optimizer = optimizer_class(net.parameters(), **optimizer_kwargs)
        
        # Train for specified number of epochs
        for epoch in range(epochs):
            # Perform one training step
            batch_size = hyperparams.get('BATCH_SIZE', 0.2)
            loss = self._one_epoch_step(net, training_data, batch_size=batch_size)
            
            if self.debug and epoch % 10 == 0:
                print(f'Worker {self.device_id} | Net #{net.net_id} | Epoch {epoch + 1} | Loss: {loss:.4f}')
                
        return net
    
    def _one_epoch_step(self, net: NN, train_data: List, batch_size: Union[int, float, None] = 0.2) -> float:
        """Perform one training epoch.
        
        Args:
            net: Network to train
            train_data: Training dataset
            batch_size: Size of minibatches
            
        Returns:
            float: Average loss for this epoch
        """
        # Implementation similar to EGD.one_epoch_step but specific to this worker
        # Move data to worker's device, train network, return loss
        # ... (adapted from EGD.one_epoch_step)
        
        # Check if train_data is empty
        if not train_data:
            raise ValueError('No training data provided')

        # Shuffle training data
        random.shuffle(train_data)

        # Track total loss
        total_loss = 0.0

        # Set model to training mode
        net.model.train()

        # Determine actual batch size
        n_samples = len(train_data)
        if batch_size is None:
            actual_batch_size = n_samples
        elif isinstance(batch_size, float):
            if not 0 < batch_size <= 1:
                raise ValueError('Batch size as percentage must be between 0 and 1')
            actual_batch_size = max(1, int(n_samples * batch_size))
        else:
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError('Batch size must be None, float between 0-1, or positive integer')
            actual_batch_size = batch_size

        # Create batches
        n_batches = (n_samples + actual_batch_size - 1) // actual_batch_size  # Ceiling division

        # Process each batch
        for i in range(n_batches):
            start_idx = i * actual_batch_size
            end_idx = min(start_idx + actual_batch_size, n_samples)
            batch = train_data[start_idx:end_idx]

            # Stack inputs and targets into batches
            batch_inputs = torch.stack([x[0] for x in batch]).to(self.device)
            batch_targets = torch.stack([x[1] for x in batch]).to(self.device)

            # Zero gradients
            net.optimizer.zero_grad()

            # Forward pass
            batch_outputs = net.model(batch_inputs)

            # Compute loss 
            loss = self._compute_loss(net, batch_targets, batch_outputs)
            total_loss += loss.item() * len(batch)  # Scale loss by batch size

            # Backward pass
            loss.backward()

            # Update weights using optimizer
            net.optimizer.step()

        # Return average loss
        return total_loss / n_samples
        
    def _compute_loss(self, net: NN, target: torch.Tensor, output: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute loss between target and output.
        
        Args:
            net: Neural network
            target: Target tensor
            output: Model output tensor
            reduction: Loss reduction method
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Move target to this worker's device
        target = target.to(self.device)
        
        # Set reduction scheme for loss function
        if hasattr(self.loss_fn, 'reduction'):
            self.loss_fn.reduction = reduction
            
        return self.loss_fn(output, target)
    
    def evaluate_net(self, net: NN, validation_data: List) -> Tuple[float, float]:
        """Evaluate network performance.
        
        Args:
            net: Network to evaluate
            validation_data: Validation dataset
            
        Returns:
            Tuple[float, float]: Error and accuracy
        """
        # Move network to this worker's device
        net.to(self.device)
        
        # Set model to evaluation mode
        net.model.eval()
        
        # Stack all inputs and targets
        inputs = torch.stack([x[0] for x in validation_data]).to(self.device)
        targets = torch.stack([x[1] for x in validation_data]).to(self.device)
        
        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            outputs = net.model(inputs)
            
            # Compute error
            errors = self._compute_loss(net, targets, outputs, reduction='none')
            avg_error = errors.mean().item()
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            _, target_classes = torch.max(targets, 1)
            accuracy = (predicted == target_classes).float().mean().item()
            
        return avg_error, accuracy

    def gc(self):
        """Explicitly trigger garbage collection on the worker."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def checkpoint(self, net: NN, path: str) -> bool:
        """Save network checkpoint to persistent storage.
        
        Args:
            net: Network to checkpoint
            path: Path to save checkpoint
            
        Returns:
            bool: Success status
        """
        try:
            net.save(f"{path}_worker{self.device_id}.pt")
            return True
        except Exception as e:
            print(f"Checkpoint error on worker {self.device_id}: {e}")
            return False


# Evolutionary Gradient Descent
class EGD:
    """Evolutionary Gradient Descent.

    Jointly optimizes neural network architecture and hyperparameters based on the
    dataset provided, the configuration, and the fitness function.

    Attributes:
        population_size (int): Size of population (min 20)
        population (Dict[int, NN]): Dict mapping indices to neural networks
        hyperparams (Dict[int, dict]): Hyperparameters for each network
        optimizer (torch.optim.Optimizer): Optimizer class to use for networks to train with.
        perfs (np.ndarray): Performance metrics for each network
        accuracies (np.ndarray): Accuracy metrics for each network
        leaderboard (np.ndarray): Network indices sorted by performance
        last_ready (np.ndarray): Last ready timestep for each network
        generations (int): Number of generations for evolutionary optimization
        epochs (int): Number of training epochs per generation
        debug (bool): Whether to print debug information
        training (List[Tuple[torch.Tensor, torch.Tensor]]): Training data (inputs, targets)
        validation (List[Tuple[torch.Tensor, torch.Tensor]]): Validation data
        loss_fn (torch.nn.Module): Loss function to use for training. A custom loss function can be provided as a torch.nn.Module.
        fitness_fn (Callable): Custom fitness function to evaluate networks.

    """

    def __init__(
            self,
            training_data: List,
            validation_data: List,
            loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
            meta_config: Dict = None,
            seed: Optional[int] = None,
            custom_fitness_fn: Optional[Callable[[float, float], float]] = None,
            distributed_gpus: Optional[Union[bool, int]] = None
    ) -> None:
        """Initialize EGD.

        Args:
            training_data: Training dataset
            validation_data: Validation dataset
            loss_fn: Loss function to use (default: CrossEntropyLoss). A custom loss function can be provided as a torch.nn.Module.
            meta_config: Configuration dictionary containing all parameters. suffixes: _RANGE, _OPTIONS, _INIT.
                        Any key that is wished to be searched over should either be appended with '_RANGE' to specify a range of for continuous values, 
                        or "_OPTIONS" to specify a list of possible categorical values.
                        Any key suffixed by "_INIT" is used as the starting point for the search.
                        If None, the search will start from the default meta_config. 
                        If not None, it should contain starting points for the hyperparameters.
            seed: Random seed for reproducibility (default: None)
            custom_fitness_fn: Optional custom fitness function
            distributed_gpus: Enable multi-GPU distribution (default: None)
                              - None: Use single GPU/CPU
                              - True: Use all available GPUs
                              - int: Use the specified number of GPUs
        """
        # Check if meta_config is provided and contains required keys
        if meta_config is None or 'LAYERS_CONFIG_INIT' not in meta_config or 'LAYERS_CONFIG_OPTIONS' not in meta_config:
            raise ValueError("meta_config must be provided and must contain 'LAYERS_CONFIG_INIT' and 'LAYERS_CONFIG_OPTIONS' keys")

        # Set random seeds if provided
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store function parameters
        self.loss_fn = loss_fn
        self.custom_fitness_fn = custom_fitness_fn
        self.distributed_gpus = distributed_gpus
        
        # Initialize configuration with default values
        default_meta_config = {
            # Population and training parameters
            'POPULATION_SIZE': 20,
            'GENERATIONS': 100,
            'EPOCHS': 200,
            'PERTS_RANGE': (0.8, 1.2),
            'MUTATION_RATE': 0.2,
            'READINESS': 20,
            'TRUNC': 0.2,
            'PERF_MULTIPLIER': 1.0,
            'COST_PENALTY': 1.0,
            'DEBUG': True,
            
            # Hyperparameter ranges
            'LEARNING_RATE_RANGE': (1e-4, 1e-1),
            'LEARNING_RATE_INIT': 0.01,
            'MOMENTUM_RANGE': (0.9, 0.99),
            'MOMENTUM_INIT': 0.95,
            'WEIGHT_DECAY_RANGE': (0.0, 0.1),
            'WEIGHT_DECAY_INIT': 0.001,
            'BATCH_SIZE_OPTIONS': [None, (0.1, 0.5), (16, 256)],
            'BATCH_SIZE_INIT': 0.2,
            'DROPOUT_RANGE': (0.0, 0.5),
            'DROPOUT_INIT': 0.2,
            
            # Network structure
            'LAYERS_CONFIG_INIT': List[ModuleConfig],
            'LAYERS_CONFIG_OPTIONS': List[List[ModuleConfig]],
            
            # Available optimizers
            'OPTIMIZER_OPTIONS': [
                optim.SGD,
                optim.Adam,
                optim.AdamW,
                optim.RMSprop,
            ],
            'OPTIMIZER_INIT': optim.Adam,
            
            # Available activations
            'ACTIVATION_OPTIONS': [
                nn.ReLU(), nn.LeakyReLU(), nn.GELU(),
                nn.Sigmoid(), nn.Tanh(), nn.SiLU(),
            ],
            'ACTIVATION_INIT': nn.ReLU(),
            
            # Available output activations
            'OUTPUT_ACTIVATION_OPTIONS': [
                None,
                nn.Softmax(dim=-1),
                nn.Sigmoid(),
                nn.Tanh(),
            ],
            'OUTPUT_ACTIVATION_INIT': nn.Softmax(dim=-1),
            
            # Available normalization layers
            'NORM_OPTIONS': [
                None,
                'batchnorm',
                'layernorm',
            ],
            'NORM_INIT': 'batchnorm',
        }
        
        # Update default config with provided values
        if meta_config is not None:
            default_meta_config.update(meta_config)
        
        # Process _INIT values to update _RANGE and _OPTIONS
        self.config = self._process_init_values(default_meta_config)
        
        # Set main parameters from config
        self.population_size = self.config['POPULATION_SIZE']
        self.generations = self.config['GENERATIONS']
        self.epochs = self.config['EPOCHS']
        self.debug = self.config['DEBUG']
        
        # Set up distributed training
        self.distributed = False
        self.num_gpus = 0
        
        # Initialize Ray for distributed training if requested
        self.ray_initialized = False
        if distributed_gpus is not None and torch.cuda.is_available():
            if isinstance(distributed_gpus, bool) and distributed_gpus:
                # Use all available GPUs
                self.num_gpus = torch.cuda.device_count()
                self.distributed = self.num_gpus > 1
            elif isinstance(distributed_gpus, int) and distributed_gpus > 1:
                # Use specified number of GPUs
                self.num_gpus = min(distributed_gpus, torch.cuda.device_count())
                self.distributed = self.num_gpus > 1
            
            if self.distributed and self.debug:
                print(f"Using Ray for distributed training across {self.num_gpus} GPUs")
            
            if self.distributed:
                # Initialize Ray with better configuration
                if not ray.is_initialized():
                    ray.init(
                        num_cpus=self.num_gpus * 2,  # Allocate CPUs for each GPU
                        include_dashboard=True,       # Enable monitoring dashboard
                        ignore_reinit_error=True,     # Handle reinitialization gracefully
                        runtime_env={"pip": ["torch", "numpy"]}  # Ensure dependencies
                    )
                    self.ray_initialized = True
                
                # Create placement group for better GPU scheduling
                pg = ray.util.placement_group(
                    [{f"GPU": 1} for _ in range(self.num_gpus)],
                    strategy="STRICT_SPREAD"
                )
                ray.get(pg.ready())
                
                # Create worker actors with placement group
                self.ray_actors = [
                    TrainingWorker.options(
                        num_gpus=1,
                        placement_group=pg,
                        placement_group_bundle_index=i
                    ).remote(
                        device_id=i, 
                        loss_fn=self.loss_fn,
                        debug=self.debug
                    )
                    for i in range(self.num_gpus)
                ]
                self.actor_pool = ActorPool(self.ray_actors)
        
        # Initialize threading components
        if self.distributed:
            self.executors = [
                ThreadPoolExecutor(max_workers=self.population_size // self.num_gpus + 1)
                for _ in range(self.num_gpus)
            ]
            self.stop_event = threading.Event()
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.population_size)
            self.stop_event = threading.Event()
        
        # Initialize data structures
        self.population = {}
        self.hyperparams = {}
        self.perfs = np.zeros(self.population_size, dtype=np.float32)
        self.accuracies = np.zeros(self.population_size, dtype=np.float32)
        self.leaderboard = np.arange(self.population_size)
        self.last_ready = np.zeros(self.population_size, dtype=np.int32)
        
        # Store datasets
        self.training = training_data
        self.validation = validation_data
        
        # Initialize tracking variables
        self.best = None
        self.most_acc = None
        self.log_path = ''
        
        # Initialize population
        self.generate_population(self.population_size)
        
        # Print debug info if enabled
        if self.debug:
            self._print_debug_info()
    
    def _process_init_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process initialization values to update ranges and options.
        
        For each key with an _INIT suffix, finds the corresponding _RANGE or _OPTIONS key
        and updates it to include the initialization value.
        
        Args:
            config: Configuration dictionary with _INIT, _RANGE, and _OPTIONS keys
            
        Returns:
            Dict[str, Any]: Updated configuration dictionary
        """
        # Create a copy of the config to avoid modifying the original
        updated_config = config.copy()
        
        # Find all keys with _INIT suffix
        init_keys = [key for key in config.keys() if key.endswith('_INIT')]
        
        for init_key in init_keys:
            # Extract the base name (without _INIT)
            base_name = init_key[:-5]  # Remove '_INIT'
            init_value = config[init_key]
            
            # Check if there's a corresponding _RANGE key
            range_key = f"{base_name}_RANGE"
            if range_key in config:
                current_range = list(config[range_key])
                
                # If init_value is outside the range, extend the range
                if init_value < current_range[0]:
                    current_range[0] = init_value
                    print(f"Extending lower bound of {range_key} to {init_value}")
                elif init_value > current_range[1]:
                    current_range[1] = init_value
                    print(f"Extending upper bound of {range_key} to {init_value}")
                
                # Update the range in the config
                updated_config[range_key] = tuple(current_range)
            
            # Check if there's a corresponding _OPTIONS key
            options_key = f"{base_name}_OPTIONS"
            if options_key in config:
                current_options = list(config[options_key])
                
                # Add init_value to options if not already present
                if init_value not in current_options:
                    current_options.append(init_value)
                    print(f"Adding {init_value} to {options_key}")
                
                # Update the options in the config
                updated_config[options_key] = current_options
                
            # If neither _RANGE nor _OPTIONS exists, create a new _OPTIONS with the init value
            if range_key not in config and options_key not in config:
                print(f"Creating new {options_key} with value {init_value}")
                updated_config[options_key] = [init_value]
        
        return updated_config
    
    def _request_stop(self):
        """Signal all threads to stop as soon as possible."""
        self.stop_event.set()

    def _print_debug_info(self) -> None:
        """Print debug information about population and dataset."""
        print('Population:', self.population)
        print('Hyperparams:', self.hyperparams)
        print('Perfs:', self.perfs)
        print('Last ready times:', self.last_ready)
        print('Training samples:', len(self.training))
        print('Validation samples:', len(self.validation))
        
        # Print shapes of features and labels
        if self.training and len(self.training) > 0:
            features, labels = self.training[0]
            print('Feature shape:', features.shape)
            print('Label shape:', labels.shape)

    def generate_population(self, population_size: int) -> None:
        """
        Generate the initial population of neural networks.
        
        Creates a population of neural networks with random architectures and hyperparameters.
        Stores the networks and their hyperparameters in the population and hyperparams 
        dictionaries.

        Args:
            population_size: Number of neural networks to generate

        Returns:
            None
            
        """
        # Generate all networks and hyperparams in parallel using list comprehension
        nets_and_params = [self.generate_net(n) for n in range(population_size)]
        # Unzip the list of tuples into separate lists
        nets, params = zip(*nets_and_params)
        # Update population and hyperparams dictionaries in bulk
        self.population.update({i: net for i, net in enumerate(nets)})
        self.hyperparams.update({i: param for i, param in enumerate(params)})


    def _get_batch_size(
            self, 
            batch_size_options: Union[None, List[Union[None, Tuple[int, int], Tuple[float, float]]]]
        ) -> Union[None, int, float]:
        """
        Determine batch size based on the provided configuration options.
        
        This helper method randomly selects one option from the provided list of batch size options,
        then processes that option to determine the final batch size value.
        
        Args:
            batch_size_options: List of configuration options for batch size, where each option can be:
                - None: No batch size specified
                - Tuple[float, float]: Range of floats to sample from (min, max)
                - Tuple[int, int]: Range of integers to sample from (min, max)
            
        Returns:
            Union[None, int, float]: The determined batch size value, which could be:
                - None if the selected option is None
                - Integer if sampling from integer range
                - Float if sampling from float range
                
        Raises:
            ValueError: If the selected batch size option has an invalid format or type
        """

        # sample from the options provided
        batch_size_option = random.choice(batch_size_options)

        # Handle None case
        if batch_size_option is None:
            batch_size = None
        # Handle tuple case
        elif isinstance(batch_size_option, tuple):
            # Check if all elements are floats
            if all(isinstance(x, float) for x in batch_size_option):
                batch_size = random.uniform(*batch_size_option)
            # Check if all elements are integers
            elif all(isinstance(x, int) for x in batch_size_option):
                batch_size = random.randint(*batch_size_option)
            # If elements are of different types, raise an error
            else:
                raise ValueError("Invalid batch size tuple format")
        elif isinstance(batch_size_option, int) or isinstance(batch_size_option, float): # any other type of batch size option is invalid
            batch_size = batch_size_option
        else:
            print(f"Invalid batch size option: {batch_size_option}")
            raise ValueError("Invalid batch size option")
        
        return batch_size
    
    def _get_layers_config(
            self, 
            layers_config_options: List[List[ModuleConfig]],
            activation_options: List[nn.Module],
            output_activation_options: List[nn.Module],
            norm_options: List[str],
        ) -> List[ModuleConfig]:
        """
        Generate a random layer configuration from the provided options.
        
        This method randomly selects a layer configuration from the provided list of options.
        # NOTE: doing it this way limits the search space to the options provided.
        # TODO: a better approach would be to sample from a distribution of layer configs 
        # TODO: and modules to mix and match. Maybe even include the registry of modules.
        # TODO: also the activation should be sampled from the activation_options.
        # TODO: also, the output activation should be sampled from the output_activation_options
        # TODO: and the norm should be sampled from the norm_options.
        # TODO: using all of these options, we can sample a random final layer configuration.
        
        Args:
            layer_configs_option: List of layer configuration options to choose from
            activation_options: List of activation options to choose from
            output_activation_options: List of output activation options to choose from
            norm_options: List of norm options to choose from

        Returns:
            List[ModuleConfig]: A randomly selected layer configuration
        """
        # simplest approach is to just sample from the options provided
        return random.choice(layers_config_options)
       
    def _get_optimizer(self, optimizer_options: List[torch.optim.Optimizer]) -> torch.optim.Optimizer:
        """
        Randomly select an optimizer from the provided options.
        
        Args:
            optimizer_options: Dictionary of optimizer options
        
        Returns:
            torch.optim.Optimizer: A randomly selected optimizer
        """
        # Randomly select optimizer from options
        return random.choice(optimizer_options)

    def generate_net(self, idx: int) -> Tuple[NN, dict]:
        """Generate a new neural network with random hyperparameters.
        
        Creates a new NN instance with randomly initialized hyperparameters within 
        the predefined ranges. The network architecture and training parameters are 
        sampled randomly.

        Args:
            idx: Unique identifier for the network

        Returns:
            Tuple containing:
              - NN: The generated neural network instance 
              - dict: The randomly generated hyperparameters
        """

        # Randomly select optimizer from options
        optimizer = self._get_optimizer(self.config['OPTIMIZER_OPTIONS'])
        # Randomly select batch size option
        batch_size = self._get_batch_size(self.config['BATCH_SIZE_OPTIONS'])
        # Select random layer config from options
        layers_config = self._get_layers_config(
            layers_config_options=self.config['LAYERS_CONFIG_OPTIONS'],
            activation_options=self.config['ACTIVATION_OPTIONS'],
            output_activation_options=self.config['OUTPUT_ACTIVATION_OPTIONS'],
            norm_options=self.config['NORM_OPTIONS'],
        )

        hyperparams = {
            'LEARNING_RATE': random.uniform(*self.config['LEARNING_RATE_RANGE']),
            'MOMENTUM': random.uniform(*self.config['MOMENTUM_RANGE']),
            'WEIGHT_DECAY': random.uniform(*self.config['WEIGHT_DECAY_RANGE']),
            'DROPOUT': random.uniform(*self.config['DROPOUT_RANGE']),
            'BATCH_SIZE': batch_size,
            'OPTIMIZER': optimizer,
            'LAYERS_CONFIG': layers_config,
        }

        # Assign to specific GPU if using distribution
        if self.distributed:
            gpu_id = idx % self.num_gpus
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = self.device
        
        # Create network with appropriate device
        net = NN(net_id=idx, hyperparams=hyperparams, debug=self.debug)
        net.to(device)

        return net, hyperparams



    def loss(
            self,
            net: NN,
            target: torch.Tensor,
            output: torch.Tensor,
            reduction: str = 'mean'
    ) -> torch.Tensor:
        """Compute the loss for the neural network.
        
        Calculates loss between target and predicted output using the specified loss function.

        Args:
            net: Neural network to compute loss for
            target: Target tensor of shape (batch_size, output_units)
            output: Model output tensor of shape (batch_size, output_units)
            reduction: Specifies the reduction to apply to the loss:
                      'none' | 'mean' | 'sum'. Default: 'mean'

        Returns:
            torch.Tensor: Scalar tensor containing the computed loss
        """
        # Move target to device and calculate main loss
        target = target.to(net.device)

        # Set reduction scheme for loss function
        if hasattr(self.loss_fn, 'reduction'):
            self.loss_fn.reduction = reduction

        loss = self.loss_fn(output, target)
        return loss

    def one_epoch_step(
            self,
            net: NN,
            train_data: List[Tuple[torch.Tensor, torch.Tensor]],
            batch_size: Union[int, float, None] = 0.2
    ) -> float:
        """Perform one training step on the given training data using minibatches.
        
        Takes training examples, splits into minibatches, performs forward and backward passes,
        and updates model parameters using SGD with momentum and weight decay.

        Args:
            net: Neural network to train
            train_data: List of (input, target) tuples where:
                - input is a tensor of shape (input_units,)
                - target is a tensor of shape (output_units,)
            batch_size: Size of minibatches to use. Can be:
                - None: use entire dataset as one batch. No stochasticity in gradient descent.
                - int > 0: use fixed batch size.
                - float between 0 and 1: use as percentage of dataset size.
                (default: 0.2) for 20% of dataset size.

        Returns:
            float: Average loss over all training examples

        Raises:
            ValueError: If train_data is empty or batch_size is invalid
        """
        # Check if stop event has been set
        if self.stop_event.is_set():
            return 0.0

        if not train_data:
            raise ValueError('No training data provided')

        # Shuffle training data
        random.shuffle(train_data)

        # Track total loss
        total_loss = 0.0

        # Set model to training mode
        net.model.train()

        # Determine actual batch size
        n_samples = len(train_data)
        if batch_size is None:
            actual_batch_size = n_samples
        elif isinstance(batch_size, float):
            if not 0 < batch_size <= 1:
                raise ValueError('Batch size as percentage must be between 0 and 1')
            actual_batch_size = max(1, int(n_samples * batch_size))
        else:
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError('Batch size must be None, float between 0-1, or positive integer')
            actual_batch_size = batch_size

        # Create batches
        n_batches = (n_samples + actual_batch_size - 1) // actual_batch_size  # Ceiling division

        # Process each batch
        for i in range(n_batches):
            # Check if stop event has been set
            if self.stop_event.is_set():
                print('Stopping training...')
                break

            start_idx = i * actual_batch_size
            end_idx = min(start_idx + actual_batch_size, n_samples)
            batch = train_data[start_idx:end_idx]

            # Stack inputs and targets into batches
            batch_inputs = torch.stack([x[0] for x in batch]).to(net.device)
            batch_targets = torch.stack([x[1] for x in batch]).to(net.device)

            # Zero gradients
            net.optimizer.zero_grad()

            # Forward pass
            batch_outputs = net.model(batch_inputs)

            # Compute loss 
            loss = self.loss(net, batch_targets, batch_outputs)
            total_loss += loss.item() * len(batch)  # Scale loss by batch size

            # Backward pass
            loss.backward()

            # Update weights using optimizer
            net.optimizer.step()

        # Return average loss
        return total_loss / n_samples

    def train_net(
            self,
            net: NN,
            train_data: List[Tuple[torch.Tensor, torch.Tensor]],
            batch_size: Union[int, float, None] = 0.2,
            epochs: int = 10
    ) -> float:
        """Train the neural network for multiple epochs.
        
        Performs multiple epochs of training using the provided training data.
        For each epoch, shuffles data and performs training steps.

        Args:
            net: Neural network instance to train
            train_data: List of (input, target) tuples for training 
            epochs: Number of training epochs (default: 10)

        Returns:
            float: Final average loss over last epoch

        Raises:
            ValueError: If train_data is empty
        """
        if not train_data:
            raise ValueError('No training data provided')

        # Train for specified number of epochs
        for epoch in range(epochs):
            # Perform one training step on all data
            avg_loss = self.one_epoch_step(net, train_data, batch_size=batch_size)

            # if self.debug:
            print(f'Epoch {epoch + 1} | Net #{net.net_id}\'s loss: {avg_loss:.4f}')

        return avg_loss

    def test_net(
            self,
            net: NN,
            test_data: List[Tuple[torch.Tensor, torch.Tensor]],
            acc_report: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """Evaluate the neural network on test data.
        
        Performs forward passes on test data and computes average error and accuracy.
        Accuracy is measured as percentage of correct predictions.
        Sets model to evaluation mode during testing.

        Args:
            net: Neural network instance to test
            test_data: List of (input, target) tuples for testing
            acc_report: Whether to also return accuracy (default: False)

        Returns:
            float: Average error on test data if acc_report is False
            Tuple[float, float]: Average error and accuracy if acc_report is True

        Raises:
            ValueError: If test_data is empty
        """
        if not test_data:
            raise ValueError('No test data provided')

        # Set model to evaluation mode
        net.model.eval()

        # Stack all inputs and targets into tensors
        inputs = torch.stack([x[0] for x in test_data])
        targets = torch.stack([x[1] for x in test_data])

        # Disable gradient computation for evaluation
        with torch.no_grad():
            # Forward pass on all data at once
            outputs = net.forward(inputs)

            # Compute total error using vectorized operations
            errors = self.loss(net, targets, outputs, reduction='none')
            avg_error = errors.mean().item()

            if acc_report:
                # Get predicted and target classes for all samples at once
                _, predicted = torch.max(outputs, 1)
                _, target_classes = torch.max(targets, 1)
                # Calculate accuracy using tensor operations
                accuracy = (predicted == target_classes).float().mean().item()
                return avg_error, accuracy

        return avg_error, 0.0

    def train_net_one_gen(self, net: NN, print_freq: int = 1) -> NN:
        """Train a neural network.
        
        Performs multiple training steps on the network using the current training data
        and hyperparameters in a separate thread. The network's optimizer and parameters 
        are updated during training. Progress is printed for each epoch.

        Args:
            n: Neural network instance to train
            print_freq: How often to print progress (default: 1)

        Returns:
            NN: The trained neural network
        """
        # Check if stop event has been set
        if self.stop_event.is_set():
            return net

        # Pre-allocate tensors and move to device
        losses = torch.zeros(self.epochs, device=net.device)

        # Train in batches using vectorized operations
        for epoch in range(self.epochs):
            # Check if stop event has been set
            if self.stop_event.is_set():
                print(f"Stop requested for net #{net.net_id}. Exiting early.")
                break

            # Get batch size from hyperparams
            batch_size = self.hyperparams[net.net_id].get('BATCH_SIZE')

            # Perform training step and store loss
            loss = self.one_epoch_step(net, self.training, batch_size=batch_size)
            losses[epoch] = loss

            # Print progress at specified frequency
            if epoch % print_freq == 0:
                print(f'Net #{net.net_id} | Epoch {epoch + 1} | Loss: {losses[epoch]:.4f}')

        return net

    async def one_gen_step(self, net: NN) -> NN:
        """Apply optimization steps to the network using Ray.
        
        Args:
            net: Neural network instance to train
            
        Returns:
            NN: The trained neural network
        """
        if not self.distributed:
            # Fall back to local training if not distributed
            return self.train_net_one_gen(net)
        
        # Serialize network for Ray
        net_id = net.net_id
        
        # Determine which GPU to use (round-robin assignment)
        gpu_id = net_id % self.num_gpus
        worker = self.ray_actors[gpu_id]
        
        # Submit training task to Ray
        train_ref = worker.train_net_one_gen.remote(
            net, 
            ray.put(self.training),  # Use ray.put to avoid repeated serialization
            self.epochs,
            self.hyperparams[net_id]
        )
        
        # Await result asynchronously
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: ray.get(train_ref))
        
        return result

    def eval_net_perf(self, net: NN) -> Tuple[float, float]:
        """
        TODO: this is a hack to get the accuracy from the loss. find a better way to handle loss and accuracy
        Evaluate the performance and accuracy of a neural network.
        
        Computes the network size and validation accuracy, then calculates an overall
        performance metric using a fitness function.

        Args:
            net: Neural network instance to evaluate

        Returns:
            Tuple[float, float]: Performance metric and accuracy
        """
        # _, accuracy = self.test_net(net, self.validation, acc_report=True)
        loss, accuracy = self.test_net(net, self.validation, acc_report=False)
        accuracy = 100 - loss
        perf = self.fitness_fn(perf=accuracy, cost=net.n_params)
        return perf, accuracy

    def fitness_fn(self, perf: float, cost: float) -> float:
        """Calculate the fitness score balancing performance and model size.

        Rewards higher performance while penalizing larger model sizes using exponential scaling.
        If a custom fitness function was provided during initialization, it will be used instead.

        Args:
            perf: Model performance. Performance should be greater as networks get better.
            cost: Model cost. Cost should be lower as networks get better.

        Returns:
            float: Fitness score (higher is better)
        """
        # Use custom fitness function if provided
        if self.custom_fitness_fn is not None:
            return self.custom_fitness_fn(perf, cost)
            
        # Default fitness function
        weighted_perf = self.config['PERF_MULTIPLIER'] * perf
        weighted_cost = self.config['COST_PENALTY'] * cost
        return weighted_perf / weighted_cost

    def copy_net_params(self, source_net: NN, target_net: NN) -> None:
        """Copy matching parameters and weights between two neural networks.
        
        Efficiently copies parameters between networks, handling cases where architectures
        partially match. For mismatched layers/units, initializes with small random values.
        
        Args:
            source_net: Network to copy parameters from
            target_net: Network to copy parameters to
            
        Returns:
            None
        """
        with torch.no_grad():
            # Get state dicts for both networks
            source_state = source_net.model.state_dict()
            target_state = target_net.model.state_dict()

            # Track which layers were successfully copied
            copied_layers = set()

            # First pass - copy exact matching layers
            for target_name, target_param in target_state.items():
                if target_name in source_state:
                    source_param = source_state[target_name]
                    if target_param.shape == source_param.shape:
                        target_param.copy_(source_param)
                        copied_layers.add(target_name)

            # Second pass - try partial copies for remaining layers
            for target_name, target_param in target_state.items():
                if target_name not in copied_layers:
                    if target_name in source_state:
                        source_param = source_state[target_name]

                        # Handle common dimension mismatches
                        try:
                            if len(target_param.shape) == len(source_param.shape):
                                # Copy what we can
                                min_dims = [min(t, s) for t, s in zip(target_param.shape, source_param.shape)]
                                slices = tuple(slice(0, d) for d in min_dims)
                                target_param[slices].copy_(source_param[slices])

                                # Initialize remaining weights
                                if target_param.shape != source_param.shape:
                                    mask = torch.ones_like(target_param, dtype=torch.bool)
                                    mask[slices] = False
                                    target_param.masked_fill_(mask, 0.0)
                                    torch.nn.init.normal_(
                                        target_param.masked_fill(~mask, 0.0),
                                        mean=0.0,
                                        std=0.01
                                    )

                                copied_layers.add(target_name)
                                continue

                        except Exception:
                            # Fall through to re-initialization
                            # print(f'Failed to copy {target_name} from {source_net.net_id} to {target_net.net_id}')
                            pass

                    # Re-initialize if copy failed
                    torch.nn.init.normal_(target_param, mean=0.0, std=0.01)

    def _copy_layers_config(self, source_hyperparams):
        """Deep copy layer configurations from source hyperparameters.
        
        Creates a completely independent copy of the layer configurations to avoid
        shared references between networks.
        
        Args:
            source_hyperparams (dict): Source hyperparameters dictionary containing layer configs
            
        Returns:
            list: A new list of ModuleConfig objects with deep-copied kwargs
        """
        
        # Return empty list if no layers config exists
        if 'LAYERS_CONFIG' not in source_hyperparams:
            print(f'No layers config found for net')
            return []
        
        # Deep copy each ModuleConfig to avoid reference issues
        return [
            ModuleConfig(
                module_type=config.module_type,
                kwargs=copy.deepcopy(config.kwargs) if config.kwargs else None
            )
            for config in source_hyperparams['LAYERS_CONFIG']
        ]

    def exploit(self, net: NN, hyperparams: dict) -> Tuple[NN, dict]:
        """Exploit better solutions via truncation selection.
        
        If the given network is among the lower-performing fraction, replaces its architecture
        and hyperparameters with those of a randomly chosen top performer.

        Args:
            net: Neural network to potentially replace
            hyperparams: Current hyperparameters dictionary

        Returns:
            Tuple[NN, dict]: Either a new network (and its hyperparameters) or the unchanged input
        """
        index = net.net_id
        bottom_threshold = 1 - self.config['TRUNC']
        bottom_idx = int(self.population_size * bottom_threshold)

        # Use numpy array indexing for faster lookup
        if index in self.leaderboard[bottom_idx:]:
            # Get top performers using array slicing
            top_count = int(self.population_size * self.config['TRUNC'])
            top_index = np.random.choice(self.leaderboard[:top_count])

            # Copy all hyperparameters from top performer
            top_hyperparams = {
                'LEARNING_RATE': self.hyperparams[top_index]['LEARNING_RATE'],
                'MOMENTUM': self.hyperparams[top_index]['MOMENTUM'],
                'WEIGHT_DECAY': self.hyperparams[top_index]['WEIGHT_DECAY'],
                'BATCH_SIZE': self.hyperparams[top_index]['BATCH_SIZE'],
                'DROPOUT': self.hyperparams[top_index]['DROPOUT'],
                'OPTIMIZER': self.hyperparams[top_index]['OPTIMIZER'],
            }
            
            # Copy layer configurations using the helper function
            top_hyperparams['LAYERS_CONFIG'] = self._copy_layers_config(self.hyperparams[top_index])

            # Create new network with copied hyperparameters
            top_net = NN(net_id=net.net_id, hyperparams=top_hyperparams, debug=self.debug)

            # Copy parameters using utility function
            self.copy_net_params(self.population[top_index], top_net)

            return top_net, top_hyperparams

        return net, hyperparams

    def explore(self, net: NN, hyperparams: dict) -> Tuple[NN, dict]:
        """Explore new hyperparameter configurations by perturbing the current ones.
        
        Randomly perturbs learning rate, momentum, weight decay, and potentially the network architecture.
        If an architectural change is made, attempts to copy weights from unchanged layers.

        Args:
            net: Neural network to explore
            hyperparams: Current hyperparameters dictionary

        Returns:
            Tuple[NN, dict]: The neural network with updated configuration and hyperparameters
        """

        mutation_rate = self.config['MUTATION_RATE']
        num_to_perturb = 3 # Number of random perturbations to generate
        num_to_mutate = 4 # Number of hyperparameters to mutate

        # Vectorized perturbation of hyperparameters using numpy
        perts = np.random.choice(self.config['PERTS_RANGE'], size=num_to_perturb)
        print(f'Perturbation factors: {perts}')
        muts = np.random.random(size=num_to_mutate)
        print(f'Mutations: {muts}')

        # Copy and perturb hyperparameters
        new_hyperparams = hyperparams.copy()
        new_hyperparams.update({
            'LEARNING_RATE': perts[0] * hyperparams['LEARNING_RATE'],
            'MOMENTUM': perts[1] * hyperparams['MOMENTUM'],
            'WEIGHT_DECAY': perts[2] * hyperparams['WEIGHT_DECAY'],
            'DROPOUT': random.uniform(*self.config['DROPOUT_RANGE']) \
                 if muts[0] < mutation_rate else hyperparams['DROPOUT'],
            'BATCH_SIZE': self._get_batch_size(
                self.config['BATCH_SIZE_OPTIONS']
            ) if muts[1] < mutation_rate else hyperparams['BATCH_SIZE'],
            'OPTIMIZER': self._get_optimizer(
                self.config['OPTIMIZER_OPTIONS']
            ) if muts[2] < mutation_rate else hyperparams['OPTIMIZER'],
            'LAYERS_CONFIG': self._get_layers_config(
                self.config['LAYERS_CONFIG_OPTIONS'],
                self.config['ACTIVATION_OPTIONS'],
                self.config['OUTPUT_ACTIVATION_OPTIONS'],
                self.config['NORM_OPTIONS'],
                # maybe some setting later to decide how to change the layers config
            ) if muts[3] < mutation_rate else hyperparams['LAYERS_CONFIG']
        })

        # since any change to the layers config will change the network architecture,
        # we need to re-initialize the new network and copy the parameters from the old
        new_net = NN(net_id=net.net_id, hyperparams=new_hyperparams, debug=self.debug)
        self.copy_net_params(net, new_net)

        return new_net, new_hyperparams

    def update_leaderboard(self) -> None:
        """
        Update the leaderboard by sorting networks based on performance.'
        
        Sorts the networks in descending order of performance (highest first).
        """
        self.leaderboard = np.argsort(-self.perfs)

    def is_ready(self, last_ready: int, timestep: int, net_id: int) -> bool:
        """Check if a network is ready for exploitation and exploration.
        
        A network is considered ready if enough epochs have passed since its last update.
        The top performing network is never eligible.

        Args:
            last_ready: Epoch when the network was last updated
            timestep: Current epoch number
            net_id: ID of the network being checked

        Returns:
            bool: True if the network is ready, False otherwise
        """
        # Early return if network is top performer
        if net_id == self.leaderboard[0]:
            return False

        # Single condition check and update
        is_ready = timestep - last_ready > self.config['READINESS']
        if is_ready:
            self.last_ready[net_id] = timestep
        return is_ready

    def is_diff(self, net1: NN, net2: NN) -> bool:
        """Check if two neural networks have different parameters.
        
        Compares the state dictionaries of the two networks using efficient tensor operations.

        Args:
            net1: First neural network to compare
            net2: Second neural network to compare

        Returns:
            bool: True if networks have any different parameters, False otherwise
        """
        # Get state dicts
        state1 = net1.model.state_dict()
        state2 = net2.model.state_dict()

        # Compare all parameters at once using torch.stack and any()
        return any(not torch.equal(state1[name], state2[name]) for name in state1.keys())

    async def train(self):
        """Train the network population using Ray for distributed computation.
        
        Returns:
            Tuple[NN, NN]: Best performing network and most accurate network
        """
        try:
            # Pre-allocate history arrays
            histories = {
                'top_acc': np.zeros(self.generations),
                'eff_acc': np.zeros(self.generations),
                'top_perf': np.zeros(self.generations),
                'top_size': np.zeros(self.generations),
                'top_lr': np.zeros(self.generations),
                'top_m': np.zeros(self.generations),
                'top_wd': np.zeros(self.generations),
                'top_dropout': np.zeros(self.generations),
                'top_batch_size': np.zeros(self.generations),
                'population_sizes': np.zeros((self.generations, self.population_size)),
                'population_accs': np.zeros((self.generations, self.population_size))
            }

            # Put training and validation data in Ray object store once
            if self.distributed:
                ray_training_data = ray.put(self.training)
                ray_validation_data = ray.put(self.validation)
            
            for gen in range(self.generations):
                print('Generation:', gen)
                
                # Check if stop event has been set
                if self.stop_event.is_set():
                    print(f"Stop requested for generation {gen}. Exiting early.")
                    break
                
                # Train all networks in parallel using Ray
                trained_nets = await asyncio.gather(*map(
                    self.one_gen_step, 
                    self.population.values()
                ),
                return_exceptions=False)
                
                # Update population with trained networks
                for net_id, net in enumerate(trained_nets):
                    self.population[net_id] = net
                    
                    # Evaluate networks using Ray workers for parallelized evaluation
                    if self.distributed:
                        # Select a worker for evaluation
                        eval_worker = self.ray_actors[net_id % self.num_gpus]
                        eval_ref = eval_worker.evaluate_net.remote(net, ray_validation_data)
                        error, accuracy = ray.get(eval_ref)
                        # Calculate performance using fitness function
                        perf = self.fitness_fn(perf=100-error, cost=net.n_params)
                    else:
                        # Use local evaluation if not distributed
                        perf, accuracy = self.eval_net_perf(net)
                    
                    self.perfs[net_id] = perf
                    self.accuracies[net_id] = accuracy
                    histories['population_sizes'][gen, net_id] = net.n_params
                    histories['population_accs'][gen, net_id] = accuracy
                
                # Update rankings
                self.update_leaderboard()
                
                # get the best and most accurate networks
                self.best = self.get_best()
                self.most_acc = self.get_most_accurate()
                
                # Update histories efficiently
                histories['top_acc'][gen] = self.most_acc[2]
                histories['eff_acc'][gen] = self.best[2]
                histories['top_perf'][gen] = self.best[1]
                histories['top_size'][gen] = self.best[0].n_params
                histories['top_lr'][gen] = self.best[3]['LEARNING_RATE']
                histories['top_m'][gen] = self.best[3]['MOMENTUM']
                histories['top_wd'][gen] = self.best[3]['WEIGHT_DECAY']
                histories['top_dropout'][gen] = self.best[3]['DROPOUT']
                histories['top_batch_size'][gen] = self.best[3]['BATCH_SIZE'] 
                
                # Print status
                print(f'Current best net perf: {self.best[1]:.2f}')
                print(f'Current best net accuracy: {self.best[2]:.2f}')
                print(f'Current best net size: {self.best[0].n_params}')
                print(f'Current best net hyperparams: {self.best[3]}')
                print(f'Current most accurate net perf: {self.most_acc[1]:.2f}')
                print(f'Current most accurate net accuracy: {self.most_acc[2]:.2f}')
                print(f'Current most accurate net size: {self.most_acc[0].n_params}')
                print(f'Current most accurate net hyperparams: {self.most_acc[3]}')
                
                # Log memory usage if using Ray
                if self.distributed and self.debug:
                    mem_info = ray.available_resources().get("memory", 0)
                    obj_store_mem = ray.available_resources().get("object_store_memory", 0)
                    print(f"Ray memory usage - Available: {mem_info/1e9:.2f} GB, "
                          f"Object store: {obj_store_mem/1e9:.2f} GB")
                
                # Explicitly garbage collect after each generation
                if self.distributed:
                    for actor in self.ray_actors:
                        actor.gc.remote()
            
            # Log histories
            utils.log_csv(
                self.log_path,
                histories,
                generations=self.generations,
                epochs=self.epochs,
                plot=True
            )

            self.best = self.get_best()
            self.most_acc = self.get_most_accurate()

            # Clean up Ray resources if we initialized Ray
            if self.ray_initialized:
                ray.shutdown()
            
            return self.best[0], self.most_acc[0]
        
        except KeyboardInterrupt:
            # Handle interruption
            print('Keyboard interrupt received. Requesting stop.')
            self._request_stop()
            
            # Clean up Ray resources if we initialized Ray
            if self.ray_initialized:
                ray.shutdown()

    def get_best(self) -> Optional[Tuple[NN, float, float, dict]]:
        """Get the neural network with the best performance from the population.

        Returns:
            Optional[Tuple[NN, float, float, dict]]: Tuple containing the best network, 
            its performance, accuracy, and hyperparameters, or None if no networks exist.
        """
        # Since self.perfs is already a numpy array, we can use numpy operations directly
        best_perf = np.max(self.perfs)

        if not self.best or self.best[1] < best_perf:
            # Get index of best performance using numpy
            index = np.argmax(self.perfs)
            # Return tuple of values at that index
            return (
                self.population[index],
                best_perf,
                self.accuracies[index],
                self.hyperparams[index]
            )

        return self.best

    def get_most_accurate(self) -> Optional[Tuple[NN, float, float, dict]]:
        """Get the neural network with the highest accuracy from the population.
        
        Returns:
            Optional[Tuple[NN, float, float, dict]]: Tuple containing the most accurate network, 
            its performance, accuracy, and hyperparameters, or None if no networks exist.
        """
        # Use numpy max which is faster than Python max
        best_acc = np.max(self.accuracies)

        if not self.most_acc or self.most_acc[2] < best_acc:
            # Get index of max accuracy using numpy
            index = np.argmax(self.accuracies)
            # Return tuple directly using index lookups
            return (
                self.population[index],
                self.perfs[index],
                best_acc,
                self.hyperparams[index]
            )

        return self.most_acc

    def __del__(self):
        """Clean up resources when the EGD instance is deleted."""
        # Shutdown Ray if we initialized it
        if hasattr(self, 'ray_initialized') and self.ray_initialized and ray.is_initialized():
            ray.shutdown()
