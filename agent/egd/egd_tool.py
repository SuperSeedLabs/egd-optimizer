from agent.egd.egd import EGD
import torch.nn as nn
import torch

egd_tool_definitions = [
    {
        "name": "run_egd",
        "description": "Run Evolutionary Gradient Descent (EGD) training on your predefined dataset. EGD optimizes both neural network architecture and hyperparameters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "population_size": {
                    "type": "integer",
                    "description": "Size of population (min 20)",
                    "minimum": 20,
                    "default": 20
                },
                "generations": {
                    "type": "integer", 
                    "description": "Number of generations to evolve",
                    "minimum": 1,
                    "default": 5
                },
                "epochs": {
                    "type": "integer",
                    "description": "Number of epochs to train each network between generations",
                    "minimum": 1,
                    "default": 200
                },
                "debug": {
                    "type": "boolean",
                    "description": "Whether to print debug information",
                    "default": True
                }
            },
            "required": ["population_size", "generations", "epochs"]
        }
    }
]

def run_egd(arguments):
    """
    Run EGD training on MNIST dataset.
    """
    if isinstance(arguments, dict):
        population_size = arguments.get("population_size", 20)
        generations = arguments.get("generations", 5) 
        epochs = arguments.get("epochs", 200)
        debug = arguments.get("debug", True)
    else:
        population_size = 20
        generations = 5
        epochs = 200
        debug = True

    try:
        # Load MNIST dataset or use a predefined one
        # Note: This would need to be implemented according to how training data is handled in your environment
        training_data = [] # This should be replaced with actual MNIST training data
        validation_data = [] # This should be replaced with actual MNIST validation data
        
        # Configure meta_config with the user-provided parameters
        meta_config = {
            "population_size": population_size,
            "generations": generations,
            "epochs": epochs,
            # Add other necessary configurations
        }
        
        # Initialize EGD with the new parameter structure
        egd = EGD(
            training_data=training_data,
            validation_data=validation_data,
            loss_fn=nn.CrossEntropyLoss(),
            meta_config=meta_config,
            seed=None,  # Optional: Set a specific seed for reproducibility
            custom_fitness_fn=None  # Optional: Custom fitness function
        )
        egd.log_path = 'logs/mnist_run.csv'

        # Train population
        best_net, most_acc = egd.train()

        # Test best network
        best_accuracy = 100 * best_net.test(egd.testing)
        best_params = best_net.num_params()

        # Test most accurate network
        most_acc_accuracy = 100 * most_acc.test(egd.testing)
        most_acc_params = most_acc.num_params()

        return {
            "tool": "run_egd",
            "status": "success",
            "attempt": f"Ran EGD with population size {population_size} for {generations} generations",
            "stdout": f"""
                Training complete.

                Best performing network:
                - Test accuracy: {best_accuracy:.2f}%
                - Parameters: {best_params}

                Most accurate network:
                - Test accuracy: {most_acc_accuracy:.2f}%
                - Parameters: {most_acc_params}
                            """,
                            "stderr": ""
                        }

    except Exception as e:
        return {
            "tool": "run_egd", 
            "status": "failure",
            "attempt": f"Tried to run EGD with population size {population_size} for {generations} generations",
            "stdout": "",
            "stderr": str(e)
        }
