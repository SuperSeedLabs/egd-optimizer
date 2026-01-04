# llm_module_generator.py
# Handles the generation, validation, and registration of novel PyTorch modules using LLMs.

import os
import asyncio
import math
import time
import json
import traceback
from typing import Tuple, List, Optional, Dict, Any, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI # Import the async client

# Assuming these are accessible from the agent structure
# If not, these imports might need adjustment based on project structure
from agent.egd.egd_registry import register_module, MODULE_REGISTRY, ModuleConfig
# Import the helper modules needed for the exec namespace
from agent.egd.egd_modules import MultiHeadAttention, FeedForward, TransformerDecoderLayer

def _validate_generated_module(
    module_name: str,
    module_code: str,
    example_inputs: Dict[str, Any],
    exec_namespace: Dict[str, Any]
) -> bool:
    """
    Performs basic validation on the generated module code.

    Tries to:
    1. Retrieve the class from the execution namespace.
    2. Instantiate the module using `example_inputs`.
    3. Perform a dummy forward pass with random integer input.
    4. Check if the output shape matches [batch_size, seq_len, vocab_size].

    Args:
        module_name: The expected name of the module class.
        module_code: The generated Python code string for the module.
        example_inputs: Dictionary containing hyperparameters needed for instantiation.
        exec_namespace: The namespace where the module code was executed.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    print(f"--- Running Basic Validation for: {module_name} ---")
    try:
        # 1. Retrieve Class
        if module_name not in exec_namespace or not isinstance(exec_namespace[module_name], type) or not issubclass(exec_namespace[module_name], nn.Module):
            print(f"Validation Error: Class '{module_name}' not found or is not a valid nn.Module in exec namespace.")
            return False
        GeneratedModuleClass: Type[nn.Module] = exec_namespace[module_name]

        # 2. Instantiate Module
        # Ensure required keys are present in example_inputs
        required_keys = ['vocab_size', 'd_model', 'num_heads', 'num_layers', 'max_seq_len', 'dropout']
        if not all(key in example_inputs for key in required_keys):
            print(f"Validation Error: Missing one or more required keys {required_keys} in example_inputs.")
            print(f"Provided keys: {list(example_inputs.keys())}")
            return False

        # Ensure numeric values are actually numbers (int/float)
        for key, value in example_inputs.items():
            if key in ['vocab_size', 'd_model', 'num_heads', 'num_layers', 'max_seq_len']:
                try:
                    example_inputs[key] = int(value)
                except (ValueError, TypeError):
                    print(f"Validation Error: Hyperparameter '{key}' must be an integer, got '{value}'.")
                    return False
            elif key == 'dropout':
                 try:
                    example_inputs[key] = float(value)
                 except (ValueError, TypeError):
                    print(f"Validation Error: Hyperparameter '{key}' must be a float, got '{value}'.")
                    return False

        print(f"Instantiating {module_name} with kwargs: {example_inputs}")
        model_instance = GeneratedModuleClass(**example_inputs)
        model_instance.eval() # Set to evaluation mode
        print(f"Successfully instantiated {module_name}.")

        # 3. Perform Dummy Forward Pass
        batch_size = 2
        seq_len = min(32, example_inputs['max_seq_len']) # Use a small sequence length for testing
        vocab_size = example_inputs['vocab_size']

        # Create dummy input tensor (long type for token indices)
        # Ensure input values are within the valid vocab range [0, vocab_size-1]
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        print(f"Performing dummy forward pass with input shape: {dummy_input.shape}")
        with torch.no_grad():
            output = model_instance(dummy_input)
        print(f"Dummy forward pass completed. Output type: {type(output)}")

        # 4. Check Output Shape
        if not isinstance(output, torch.Tensor):
            print(f"Validation Error: Output is not a torch.Tensor, but {type(output)}.")
            return False

        expected_shape = (batch_size, seq_len, vocab_size)
        if output.shape != expected_shape:
            print(f"Validation Error: Output shape mismatch. Expected {expected_shape}, Got {output.shape}")
            return False

        print(f"Output shape {output.shape} matches expected shape {expected_shape}.")
        print(f"--- Basic Validation PASSED for: {module_name} ---")
        return True

    except Exception as e:
        print(f"Validation Error: Exception during validation for '{module_name}': {e}")
        print("--- Offending Code --- ")
        print(module_code)
        print("--- End Offending Code --- ")
        traceback.print_exc()
        print(f"--- Basic Validation FAILED for: {module_name} ---")
        return False


# Function to generate, validate, and optionally correct a single module (NOW ASYNC)
async def generate_validate_correct_module_single(
    client: AsyncOpenAI, # Pass async client
    existing_transformer_example: str,
    max_correction_attempts: int = 2 # Number of times to try correcting a faulty generation
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Generates a novel PyTorch module using OpenAI (async), validates its basic functionality,
    attempts to self-correct using the LLM if validation fails, and registers it locally.

    Args:
        client: An instance of the AsyncOpenAI client.
        existing_transformer_example: String containing the code of TransformerLM as a structural guide.
        max_correction_attempts: Maximum number of times to ask the LLM to fix the code.

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: A tuple containing the final registered module name
            and its configuration kwargs if successful (generation + validation + registration),
            None otherwise.
    """
    # API key check should happen before calling this function

    # Initial Generation Attempt
    print(f"--- Initial Generation Attempt --- ")
    module_code: Optional[str] = None
    suggested_name: Optional[str] = None
    final_name: Optional[str] = None
    example_inputs: Optional[Dict[str, Any]] = None
    original_code: Optional[str] = None # Store the first generated code

    correction_prompt = f"""
    Act as a world-class AI researcher specializing in novel transformer architectures.
    Your task is to design a *novel* and *potentially groundbreaking* PyTorch neural network module
    class for language modeling, suitable for datasets like BabyLM. MAKE SURE THERE ARE NO BUGS.

    **Crucially, suggest a descriptive, unique, CamelCase Python class name for your main creation.**

    While aiming for innovation, the module MUST be functional and adhere to the following structure
    inspired by the provided `TransformerLM` example:

    1.  **Inherit:** Must inherit from `torch.nn.Module`.
    2.  **Initialization (`__init__`)**:
        *   Accept `**kwargs`.
        *   Extract hyperparameters: `vocab_size`, `d_model`, `num_heads`, `num_layers`, `max_seq_len`, `dropout`. Use typical values (see example JSON) as defaults if possible. Document clearly.
        *   Store `max_seq_len` as `self.block_size`.
        *   Calculate and store total parameters in `self.n_params`.
    3.  **Forward Pass (`forward`)**:
        *   Input shape: `[batch_size, sequence_length]`.
        *   Output shape: `[batch_size, sequence_length, vocab_size]` (logits).
    4.  **Generation (`generate`)**: Implement the `generate` method exactly as shown in the example for text generation.
    5.  **Sub-modules**:
        *   If you design novel helper sub-modules (e.g., custom attention, FFN layers), **define their classes *directly inside* the main module class you are generating.**
        *   You may also use the standard helper classes provided globally, which are already registered: `MultiHeadAttention`, `FeedForward`, `TransformerDecoderLayer`.
        *   **IMPORTANT**: For any *custom* helper sub-module class you define inside the main class, you **MUST** add the `@register_module('UniqueSubModuleName')` decorator directly above its class definition. Choose a unique, descriptive name for 'UniqueSubModuleName'.
    6.  **Documentation**: Include comprehensive docstrings and type hints.
    7.  **Code Style**: Do NOT include top-level imports (`import torch`, etc.) in the code block. Assume `torch`, `nn`, `F`, `math`, `@register_module`, and the standard helper classes are available. The main class name in the code MUST match your suggested name.

    **Architectural Freedom:** You have freedom to propose novel variations. This could involve:
    *   Different attention mechanisms (e.g., sparse, linear, gated).
    *   Novel feed-forward network designs (e.g., alternative activations, parallel paths).
    *   Changes to normalization or residual connections (e.g., Pre-LN, skip connections).
    *   Different ways of incorporating positional information.
    *   Other architectural innovations you deem promising for language modeling.

    **Output Format:** Return ONLY the following JSON structure:
    {{
        "suggested_module_name": "YourProposedCamelCaseName",
        "module_code": "... full Python code for the YourProposedCamelCaseName class ONLY, including any custom helper classes defined *inside* it, with `@register_module` decorators applied to custom helpers...",
        "description": "... concise description of the module's novel aspects and functionality ...",
        "example_inputs": {{
            "vocab_size": 50257,
            "d_model": 768,
            "num_heads": 12,
            "num_layers": 6,
            "max_seq_len": 1024,
            "dropout": 0.1
            // Add other key parameters with typical values if your design introduces them
        }}
    }}

    Reference `TransformerLM` Structure (for API compatibility, not architectural limitation):
    ```python
    {existing_transformer_example}
    ```
    """

    attempt = 0
    while attempt <= max_correction_attempts:
        task_id = id(asyncio.current_task()) if asyncio.current_task() else "main"
        print(f"[Task {task_id}] --- Generation/Correction Attempt {attempt + 1} / {max_correction_attempts + 1} for '{suggested_name or 'New Module'}'---")
        current_prompt = correction_prompt # Start with the base prompt

        try:
            # --- LLM Call (Async) ---
            print(f"[Task {task_id}] Calling OpenAI API (Attempt {attempt + 1})...")
            response = await client.chat.completions.create( # Use await and the async client
                model="o3",
                messages=[
                    {"role": "system", "content": "You are an AI researcher designing and debugging PyTorch transformer modules."},
                    {"role": "user", "content": current_prompt}
                ],
                response_format={"type": "json_object"},
            )
            response_content = response.choices[0].message.content.strip()
            response_data = json.loads(response_content)
            print(f"[Task {task_id}] API call successful. Processing response...")

            # --- Extract Data ---
            new_suggested_name = response_data.get("suggested_module_name")
            new_module_code = response_data.get("module_code")
            new_example_inputs = response_data.get("example_inputs", {})

            if not new_suggested_name or not new_module_code or not new_example_inputs:
                print(f"[Task {task_id}] Error: LLM response missing required fields. Aborting this attempt.")
                if attempt == max_correction_attempts: return None
                attempt += 1
                continue

            if suggested_name is None: suggested_name = new_suggested_name
            if original_code is None: original_code = new_module_code

            module_code = new_module_code
            example_inputs = new_example_inputs

            # --- Name Disambiguation ---
            if final_name is None:
                final_name = suggested_name
                suffix_counter = 1
                # CRITICAL: Access to MODULE_REGISTRY needs to be thread-safe/process-safe
                # if running truly parallel (not just concurrent). For asyncio concurrency,
                # direct access might be okay, but locking is safer if modifications happen.
                # Using a lock if modifying registry within the loop.
                # NOTE: Assuming asyncio concurrency here, potential race condition if registry
                # check/update isn't atomic across tasks. A lock would be needed for true safety.
                while final_name in MODULE_REGISTRY:
                    suffix_counter += 1
                    final_name = f"{suggested_name}_v{suffix_counter}"
                if final_name != suggested_name:
                    print(f"[Task {task_id}] Warning: Module name '{suggested_name}' exists. Using '{final_name}'.")

            # --- Code Preparation ---
            current_code_to_exec = module_code
            if suggested_name != final_name:
                current_code_to_exec = current_code_to_exec.replace(f"class {suggested_name}(", f"class {final_name}(", 1)
            if not f"class {final_name}(" in current_code_to_exec:
                 print(f"[Task {task_id}] Warning: Class definition for '{final_name}' not found after potential rename.")

            print(f"[Task {task_id}] Attempting execution & validation for: {final_name}")
            exec_namespace = {
                'torch': torch, 'nn': nn, 'F': F, 'math': math,
                'MultiHeadAttention': MultiHeadAttention, 'FeedForward': FeedForward,
                'TransformerDecoderLayer': TransformerDecoderLayer, 'register_module': register_module
            }

            # --- Execute Code ---
            try:
                exec(current_code_to_exec, exec_namespace)
                print(f"[Task {task_id}] Code executed successfully for {final_name}.")
            except Exception as exec_error:
                 print(f"[Task {task_id}] Error executing generated code for {final_name}: {exec_error}")
                 error_traceback = traceback.format_exc()
                 print(error_traceback)
                 if attempt < max_correction_attempts:
                     print(f"[Task {task_id}] Preparing correction prompt...")
                     correction_prompt = _create_correction_prompt(
                         original_code if original_code else "Code not available",
                         error_traceback,
                         existing_transformer_example,
                         suggested_name
                     )
                     attempt += 1
                     continue
                 else:
                     print(f"[Task {task_id}] Max correction attempts reached after execution error. Giving up on {final_name}.")
                     return None

            # --- Validate Code ---
            # Validation itself is synchronous CPU-bound code, okay to run directly
            validation_passed = _validate_generated_module(final_name, current_code_to_exec, example_inputs, exec_namespace)

            if validation_passed:
                print(f"[Task {task_id}] Validation PASSED for {final_name}.")
                # --- Final Registration (Needs thread/process safety) ---
                # Again, potential race condition here without locking MODULE_REGISTRY access
                if final_name in exec_namespace and isinstance(exec_namespace[final_name], type) and issubclass(exec_namespace[final_name], nn.Module):
                    generated_module_class = exec_namespace[final_name]
                    # Use final_name which is guaranteed unique for this task *run*
                    MODULE_REGISTRY[final_name] = generated_module_class
                    print(f"[Task {task_id}] Module {final_name} ensured in registry.")

                    # --- Print Source Code (final validated version) ---
                    print(f"\n--- Source Code for validated module: {final_name} ---")
                    print("```python")
                    print(module_code.strip()) # Print the code that passed validation
                    print("```")
                    print(f"--- End Source Code for {final_name} ---\n")
                    # --- End Print Source Code ---

                    return final_name, example_inputs # Success!
                else:
                     print(f"[Task {task_id}] Error: Class {final_name} not found after validation. Internal error.")
                     # Attempt cleanup before failing
                     if final_name in MODULE_REGISTRY: del MODULE_REGISTRY[final_name]
                     return None

            else: # Validation failed
                print(f"[Task {task_id}] Validation FAILED for {final_name}.")
                # --- Prepare for Correction --- (Cleanup registry if needed)
                # Needs thread/process safety if multiple tasks might delete same *potential* name
                if final_name in MODULE_REGISTRY:
                     print(f"[Task {task_id}] Removing {final_name} from registry before correction attempt.")
                     try: del MODULE_REGISTRY[final_name]
                     except KeyError: pass # Already removed by another task?

                if attempt < max_correction_attempts:
                    print(f"[Task {task_id}] Preparing correction prompt based on validation failure...")
                    error_message = f"Basic validation failed for module {final_name}. Check __init__, forward shape, etc."
                    correction_prompt = _create_correction_prompt(
                         current_code_to_exec,
                         error_message,
                         existing_transformer_example,
                         suggested_name
                     )
                    attempt += 1
                    continue
                else:
                    print(f"[Task {task_id}] Max correction attempts reached after validation failure. Giving up on {final_name}.")
                    return None # Failed validation after max attempts

        except json.JSONDecodeError as e:
            print(f"[Task {task_id}] Error decoding JSON: {e}")
            if attempt < max_correction_attempts: attempt += 1; continue
            else: return None
        except openai.APIError as e:
             print(f"[Task {task_id}] OpenAI API Error: {e}")
             if attempt < max_correction_attempts: attempt += 1; continue
             else: return None
        except Exception as e:
            print(f"[Task {task_id}] Unexpected error during attempt {attempt + 1} for '{suggested_name or 'New'}': {e}")
            traceback.print_exc()
            if attempt < max_correction_attempts: attempt += 1; continue
            else: return None

    print(f"[Task {task_id}] Exited generation/correction loop unexpectedly for '{suggested_name or 'New'}'.")
    return None


def _create_correction_prompt(
    faulty_code: str,
    error_message: str,
    existing_transformer_example: str,
    original_suggested_name: str
) -> str:
    """
    Creates a prompt for the LLM to correct the previously generated code.

    Args:
        faulty_code: The Python code string that caused an error or failed validation.
        error_message: The error traceback or a description of the validation failure.
        existing_transformer_example: The reference TransformerLM code.
        original_suggested_name: The name the LLM first suggested for the module.

    Returns:
        str: The prompt string asking the LLM to fix the code.
    """
    print("--- Creating Correction Prompt ---")
    prompt = f"""
You previously generated the following PyTorch module code, intended to be named '{original_suggested_name}'.
However, it produced an error or failed validation.

**Faulty Code:**
```python
{faulty_code}
```

**Error/Validation Failure:**
```
{error_message}
```

**Task:** Please fix the `module_code` below to resolve the error and ensure it meets all the original requirements (inherits nn.Module, correct __init__, forward, generate methods, handles kwargs, defines internal submodules with @register_module if custom, etc.). Pay close attention to the error message.

**Original Requirements Reminder:**
1.  **Inherit:** Must inherit from `torch.nn.Module`.
2.  **Initialization (`__init__`)**: Accept `**kwargs`, extract standard hyperparameters, store `max_seq_len` as `self.block_size`, calculate `self.n_params`.
3.  **Forward Pass (`forward`)**: Input `[batch, seq]`, Output `[batch, seq, vocab_size]`.
4.  **Generation (`generate`)**: Implement the standard `generate` method.
5.  **Sub-modules**: Define custom helpers *inside* the main class, decorate with `@register_module('UniqueSubModuleName')`. Standard helpers (`MultiHeadAttention`, `FeedForward`, `TransformerDecoderLayer`) are available.
6.  **Code Style**: No top-level imports. Assume `torch`, `nn`, `F`, `math`, `@register_module` are available.

**Output Format:** Return ONLY the corrected JSON structure. **Crucially, use the SAME `suggested_module_name` ('{original_suggested_name}') as before.** Ensure the `module_code` is the complete, corrected code for the main class, including any internal helper classes.
{{
    "suggested_module_name": "{original_suggested_name}",
    "module_code": "... full CORRECTED Python code for the {original_suggested_name} class ONLY ...",
    "description": "... concise description of the module's novel aspects and functionality (can be same as before or updated) ...",
    "example_inputs": {{ // Keep the original example inputs unless the fix requires changing them
        "vocab_size": 50257,
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 6,
        "max_seq_len": 1024,
        "dropout": 0.1
        // Add other key parameters if needed
    }}
}}

Reference `TransformerLM` Structure:
```python
{existing_transformer_example}
```
    """
    return prompt


# Parallel Generation Function (Implementation)
async def generate_validated_modules_parallel(
    num_modules: int,
    existing_transformer_example: str,
    max_correction_attempts: int = 2
) -> List[ModuleConfig]:
    """Generates multiple novel modules in parallel using asyncio and OpenAI.

    Handles validation and self-correction attempts for each module concurrently.

    Args:
        num_modules: The desired number of valid modules to generate.
        existing_transformer_example: String source code of the reference TransformerLM.
        max_correction_attempts: Max correction attempts per module.

    Returns:
        List[ModuleConfig]: A list of ModuleConfig objects for the successfully
                          generated and validated modules.
    """
    print(f"\n--- Starting Parallel Generation for {num_modules} modules ---")
    load_dotenv()
    api_key = os.getenv("OPENAI")
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Cannot generate modules.")
        return []

    # Initialize the async client
    client = AsyncOpenAI(api_key=api_key)

    tasks = []
    for i in range(num_modules):
        # Create an asyncio task for each generation attempt
        task = asyncio.create_task(
            generate_validate_correct_module_single(
                client=client,
                existing_transformer_example=existing_transformer_example,
                max_correction_attempts=max_correction_attempts
            ),
            name=f"ModuleGenTask-{i+1}" # Give tasks names for easier debugging
        )
        tasks.append(task)

    print(f"Created {len(tasks)} generation tasks. Waiting for completion...")

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True) # Capture exceptions

    print(f"--- Parallel Generation Complete --- Processing {len(results)} results ---")

    successful_modules: List[ModuleConfig] = []
    for i, result in enumerate(results):
        task_name = tasks[i].get_name() if hasattr(tasks[i], 'get_name') else f"Task-{i+1}"
        if isinstance(result, Exception):
            print(f"Error in {task_name}: {result}")
        elif result is not None:
            module_name, module_kwargs = result
            print(f"Success from {task_name}: Module '{module_name}' generated and validated.")
            successful_modules.append(ModuleConfig(module_type=module_name, kwargs=module_kwargs))
        else:
            print(f"Failure from {task_name}: Module generation/validation did not succeed.")

    print(f"Successfully generated and validated {len(successful_modules)} / {num_modules} modules.")
    return successful_modules 