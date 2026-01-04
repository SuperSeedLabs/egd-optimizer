import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np


def corrupt_data(data: list, classes: list, percent: float) -> list:
    """Corrupt class labels in training data by randomly changing labels.
    
    Takes a dataset and randomly changes the class labels for a specified percentage
    of examples to different valid class labels. This is useful for testing model
    robustness to noisy/corrupted training data.
    
    Args:
        data: List of training examples, where each example's last element is the class label
        classes: List of valid class labels that can be assigned
        percent: Float between 0 and 1 indicating what fraction of examples to corrupt
        
    Returns:
        List containing the corrupted dataset with modified class labels
    """
    # Calculate number of examples to corrupt based on percentage
    num_examples = len(data)
    num_examples_to_corrupt = int(percent * num_examples)
    
    # Randomly select indices of examples to corrupt
    corrupt_elements = random.sample(range(num_examples), num_examples_to_corrupt)

    # Corrupt the selected examples by changing their class labels
    for e in corrupt_elements:
        # Get the current class label
        correct_label = data[e][-1]
        
        # Select a random different class label
        random_class = random.choice(classes)
        while random_class == correct_label:
            random_class = random.choice(classes)
    
        # Replace the original label with the corrupted one
        data[e][-1] = random_class
        
    return data

def format_csv_row(gen: int, epoch: int, metrics: list) -> str:
    """Format a single row of CSV data.
    
    Args:
        gen: Generation number
        epoch: Epoch number within generation
        metrics: List of metric values to include
        
    Returns:
        Comma-separated string containing the formatted row
    """
    row = f"{gen},{epoch}"
    for metric in metrics:
        row += f",{metric}"
    return row

def parse_csv_row(row: str) -> tuple[int, int, list]:
    """Parse a CSV row into generation, epoch and metrics.
    
    Args:
        row: String containing comma-separated values
        
    Returns:
        Tuple containing:
            - Generation number (int)
            - Epoch number (int) 
            - List of metric values (float)
    """
    values = row.strip().split(',')
    gen = int(values[0])
    epoch = int(values[1])
    metrics = [float(x) for x in values[2:]]
    return gen, epoch, metrics


def print_elapsed_time(training_time: float) -> None:
    """Print elapsed time in days, hours, minutes, seconds format.
    
    Args:
        training_time: Time in seconds to format and print
    """
    days = training_time // (24 * 3600)
    hours = (training_time % (24 * 3600)) // 3600 
    minutes = (training_time % 3600) // 60
    seconds = training_time % 60

    print('\nPopulation Based Training complete\n')
    print(f'Training completed in:')
    if days > 0:
        print(f'{int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds')
    elif hours > 0:
        print(f'{int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds')
    elif minutes > 0:
        print(f'{int(minutes)} minutes, {seconds:.2f} seconds')
    else:
        print(f'{seconds:.2f} seconds')

def log_csv(
        path: str, 
        histories: dict,
        generations: int, 
        epochs: int, 
        plot: bool = False) -> None:
    """Log training histories to a CSV file and optionally plot metrics.
    
    Creates necessary directories if they don't exist and writes training metrics
    to a CSV file with generation and epoch numbers. Headers are derived from 
    histories dictionary keys. Can also generate plots of each metric over epochs,
    marking generation boundaries.
    
    Args:
        path: Path to the output CSV file
        histories: Dictionary containing metric histories to log (per generation)
        generations: Number of generations run
        epochs: Number of epochs per generation
        plot: Whether to generate plots of metrics (default: False)
        
    Returns:
        None
    """
    # Get timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base filename with timestamp
    base_path = os.path.splitext(path)[0]
    base_name = os.path.basename(base_path)
    timestamped_base = f"{base_name}_{timestamp}"
    
    # Update CSV path with timestamp
    csv_path = os.path.join(os.path.dirname(path), f"{timestamped_base}.csv")
    
    # Create directory path if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Get headers from histories dict, excluding population arrays
    headers = ['gen', 'epoch'] + [
        key for key in histories.keys() if key not in ['population_sizes', 'population_accs']]
    
    # Add headers for each network's size and accuracy
    for i in range(len(histories['population_sizes'][0])):
        headers.extend([f'net{i}_size', f'net{i}_acc'])
    
    # Open file and write data
    try:
        with open(csv_path, 'w') as f:
            # Write CSV headers
            f.write(','.join(headers) + '\n')
            
            # Write each generation's metrics
            for gen in range(generations):
                # For each epoch in the generation
                for epoch in range(epochs):
                    # Get scalar metrics for this generation
                    metrics = [histories[key][gen] for key in headers[2:] 
                             if key in histories]
                    
                    # Add size and accuracy for each network
                    for net_idx in range(len(histories['population_sizes'][0])):
                        metrics.extend([
                            histories['population_sizes'][gen][net_idx],
                            histories['population_accs'][gen][net_idx]
                        ])
                    
                    # Format and write row
                    row = format_csv_row(gen, epoch, metrics)
                    f.write(row + '\n')
                
        if plot:
            # Create plots directory
            plots_dir = os.path.join(os.path.dirname(path), 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Create epoch points for x-axis (all epochs)
            epoch_points = range(generations * epochs)
            
            # Plot each metric (excluding population arrays)
            for header in headers[2:]:  # Skip gen and epoch columns
                if header.startswith('net'): continue # Skip individual network metrics
                    
                plt.figure(figsize=(10, 6))
                
                # Expand generation values across their epochs
                epoch_values = []
                for gen_value in histories[header]:
                    # Repeat each generation's value for all its epochs
                    epoch_values.extend([gen_value] * epochs)
                
                plt.plot(epoch_points, epoch_values)
                
                # Add vertical lines for generation boundaries
                for gen in range(generations):
                    plt.axvline(x=gen*epochs, color='gray', linestyle='--', alpha=0.5)
                
                plt.title(f'{header} vs Epochs (with Generation Boundaries)')
                plt.xlabel('Epochs')
                plt.ylabel(header)
                plt.grid(True)
                
                # Add generation labels
                for gen in range(generations):
                    plt.text(gen*epochs, plt.ylim()[0], f'Gen {gen}', 
                            rotation=90, verticalalignment='bottom')
                
                # Save plot
                plot_path = os.path.join(plots_dir, f"{timestamped_base}_{header}.png")
                plt.savefig(plot_path)
                plt.close()

            # Create size vs accuracy scatter plot
            plt.figure(figsize=(10, 6))
            
            # Get final sizes and accuracies for each network
            final_sizes = histories['population_sizes'][-1]  # Last generation's sizes
            final_accs = histories['population_accs'][-1]    # Last generation's accuracies
            
            # Create scatter plot
            plt.scatter(final_sizes, final_accs)
            
            # Add network ID labels to each point
            for i, (size, acc) in enumerate(zip(final_sizes, final_accs)):
                plt.annotate(f'Net {i}', (size, acc), 
                           xytext=(5, 5), textcoords='offset points')
            
            plt.title('Network Size vs Accuracy (Final Generation)')
            plt.xlabel('Network Size (Parameters)')
            plt.ylabel('Accuracy')
            plt.grid(True)
            
            # Save size vs accuracy plot
            plot_path = os.path.join(plots_dir, f"{timestamped_base}_size_vs_accuracy.png")
            plt.savefig(plot_path)
            plt.close()
                
    except IOError as e:
        print(f"Error writing to {csv_path}: {e}")
        raise


def show_results_streamlit(
        csv_path: str = "C:/Users/the_3/Documents/github/ARG-Main-Agent/logs/mnist_20250214_152649.csv"):
    """Display training results in a Streamlit interface.
    
    Creates an interactive dashboard showing training metrics and plots using Streamlit.
    Allows toggling between viewing results per generation or expanded across epochs.
    
    Args:
        csv_path: Path to the CSV file containing training metrics
        
    Returns:
        None
    """
    
    st.title('EGD Training Results Dashboard')
    
    # Read CSV data
    try:
        df = pd.read_csv(csv_path)
        
        # Get total number of rows and generations for x-axis
        total_rows = len(df)
        generations = df['gen'].max() + 1
        x_values = np.arange(total_rows)
        x_label = 'Training Step'
        values_dict = {col: df[col].values for col in df.columns if col != 'gen'}
        
        # Display summary statistics
        st.header('Summary Statistics')
        st.dataframe(df.describe())
        
        # Display raw data table
        st.header('Raw Data')
        st.dataframe(df)

        # Plot metrics
        st.header('Training Metrics')
        
        # Create tabs for different metric categories
        tab1, tab2, tab3, tab4 = st.tabs(['Accuracy Metrics', 'Network Metrics', 'Hyperparameters', 'Size vs Accuracy'])
        
        # Helper function to add generation markers
        def add_generation_markers(ax):
            steps_per_gen = total_rows // generations
            for gen in range(generations):
                ax.axvline(x=gen*steps_per_gen, color='gray', linestyle='--', alpha=0.5)
                ax.text(gen*steps_per_gen, ax.get_ylim()[0], f'Gen {gen}',
                       rotation=90, verticalalignment='bottom')
        
        with tab1:
            st.subheader('Accuracy Metrics')
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_values, values_dict['top_acc'], label='Top Accuracy')
            ax.plot(x_values, values_dict['eff_acc'], label='Effective Accuracy')
            ax.set_xlabel(x_label)
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True)
            add_generation_markers(ax)
            st.pyplot(fig)
            
            # Add line chart using Streamlit
            st.line_chart(df[['top_acc', 'eff_acc']])
            
        with tab2:
            st.subheader('Network Performance')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            ax1.plot(x_values, values_dict['top_perf'], color='green')
            ax1.set_ylabel('Performance')
            ax1.grid(True)
            add_generation_markers(ax1)
            
            ax2.plot(x_values, values_dict['top_size'], color='orange')
            ax2.set_ylabel('Network Size')
            ax2.set_xlabel(x_label)
            ax2.grid(True)
            add_generation_markers(ax2)
            st.pyplot(fig)
            
            # Add line charts using Streamlit
            st.line_chart(df['top_perf'])
            st.line_chart(df['top_size'])
            
        with tab3:
            st.subheader('Hyperparameter Evolution')
            
            # Create two rows of hyperparameter plots
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 15))
            
            # Learning rate
            ax1.plot(x_values, values_dict['top_lr'], color='blue')
            ax1.set_ylabel('Learning Rate')
            ax1.grid(True)
            add_generation_markers(ax1)
            
            # Momentum
            ax2.plot(x_values, values_dict['top_m'], color='purple')
            ax2.set_ylabel('Momentum') 
            ax2.grid(True)
            add_generation_markers(ax2)
            
            # Weight decay
            ax3.plot(x_values, values_dict['top_wd'], color='red')
            ax3.set_ylabel('Weight Decay')
            ax3.grid(True)
            add_generation_markers(ax3)
            
            # Dropout
            ax4.plot(x_values, values_dict['top_dropout'], color='green')
            ax4.set_ylabel('Dropout')
            ax4.grid(True)
            add_generation_markers(ax4)
            
            # Batch size
            ax5.plot(x_values, values_dict['top_batch_size'], color='orange')
            ax5.set_ylabel('Batch Size')
            ax5.set_xlabel(x_label)
            ax5.grid(True)
            add_generation_markers(ax5)
            
            # Performance
            ax6.plot(x_values, values_dict['top_perf'], color='brown')
            ax6.set_ylabel('Performance')
            ax6.set_xlabel(x_label)
            ax6.grid(True)
            add_generation_markers(ax6)
            
            st.pyplot(fig)
            
            # Add line charts using Streamlit in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.line_chart(df['top_lr'], use_container_width=True)
                st.line_chart(df['top_dropout'], use_container_width=True)
            with col2:
                st.line_chart(df['top_m'], use_container_width=True)
                st.line_chart(df['top_batch_size'], use_container_width=True)
            with col3:
                st.line_chart(df['top_wd'], use_container_width=True)
                st.line_chart(df['top_perf'], use_container_width=True)
                
        with tab4:
            st.subheader('Network Size vs Accuracy Analysis')
            
            # Get network specific columns
            net_sizes = [f'net{i}_size' for i in range(10)]
            net_accs = [f'net{i}_acc' for i in range(10)]
            
            # Create scatter plot of final generation
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get final generation values for all rows
            final_sizes = df[net_sizes].iloc[-1:].values.flatten()  # Get last row as 1D array
            final_accs = df[net_accs].iloc[-1:].values.flatten()   # Get last row as 1D array
            
            # Create scatter plot
            scatter = ax.scatter(final_sizes, final_accs)
            
            # Add labels for each point
            for i, (size, acc) in enumerate(zip(final_sizes, final_accs)):
                ax.annotate(f'Net {i}', (size, acc), 
                          xytext=(5, 5), textcoords='offset points')
            
            ax.set_title('Network Size vs Accuracy (Final Generation)')
            ax.set_xlabel('Network Size (Parameters)')
            ax.set_ylabel('Accuracy')
            ax.grid(True)
            
            st.pyplot(fig)
            
            # Add interactive scatter plot using Streamlit
            size_acc_df = pd.DataFrame({
                'Size': final_sizes,
                'Accuracy': final_accs,
                'Network': [f'Net {i}' for i in range(len(final_sizes))]
            })
            
            st.scatter_chart(
                data=size_acc_df,
                x='Size',
                y='Accuracy',
                color='Network'
            )
            
        # Add download button for CSV
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name="training_results.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")

if __name__ == "__main__":
    show_results_streamlit()
    # Run with: streamlit run Coder/agent/tools/egd/utils.py
