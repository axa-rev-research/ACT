"""
demo_helper.py
==============

Helper functions for ACT demo notebook.
Provides a clean, sklearn-like interface for the ACT framework.

Usage:
    from helper_demo import load_dataset, train_model, evaluate_model, plot_tree
"""

import sys
import os
import numpy as np

# Add local textgrad to path
local_textgrad_path = os.path.join(os.path.dirname(__file__), '../textgrad')  # Assumes textgrad folder is in same directory
if os.path.exists(local_textgrad_path):
    sys.path.insert(0, os.path.dirname(__file__))
    print(f"[INFO] Using local TextGrad from: {local_textgrad_path}")
else:
    print("[WARNING] Local TextGrad not found, using installed version")

import textgrad as tg
from textgrad.tasks import load_task

import demo_functions as fn
import demo_prompts as fp

def setup_engine(model_string, api_key=None, **engine_kwargs):
    """
    Setup any LLM engine supported by TextGrad.
    
    Args:
        model_string: Model identifier following TextGrad conventions:
            - OpenAI: "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"
            - Azure: "azure-<deployment-name>" (requires AZURE_* env vars)
            - Claude: "claude-3-5-sonnet-20240620", "opus", "haiku"
            - Ollama: "ollama-gemma3:4b", "ollama-llama3"
            - Other: See TextGrad documentation
        api_key: API key for the service. If None, reads from environment.
                 For OpenAI: sets OPENAI_API_KEY
                 For Azure: sets AZURE_API_KEY and AZURE_OPENAI_API_KEY
                 For Anthropic: sets ANTHROPIC_API_KEY
        **engine_kwargs: Additional arguments to pass to tg.get_engine()
    
    Returns:
        Configured TextGrad engine
    
    Example:
        # OpenAI
        engine = setup_engine("gpt-4o", api_key="sk-...")
        
        # Azure (requires env vars: AZURE_OPENAI_ENDPOINT, OPENAI_API_VERSION, OPENAI_MODEL_NAME)
        engine = setup_engine("azure-gpt-4o")
        
        # Claude
        engine = setup_engine("claude-3-5-sonnet-20240620", api_key="sk-ant-...")
        
        # Ollama (local)
        engine = setup_engine("ollama-gemma3:4b")
    """
    # Set API key in environment if provided
    if api_key:
        if model_string.startswith("azure-"):
            os.environ['AZURE_API_KEY'] = api_key
            os.environ['AZURE_OPENAI_API_KEY'] = api_key
            os.environ['OPENAI_API_KEY'] = api_key  # Some versions need this too
        elif "claude" in model_string or model_string in ["opus", "haiku", "sonnet", "sonnet-3.5"]:
            os.environ['ANTHROPIC_API_KEY'] = api_key
        elif "gpt" in model_string:
            os.environ['OPENAI_API_KEY'] = api_key
        else:
            # Generic fallback - set OPENAI_API_KEY as it's commonly used
            os.environ['OPENAI_API_KEY'] = api_key
    
    # Special handling for Azure - validate required env vars
    if model_string.startswith("azure-"):
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'OPENAI_API_VERSION']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(
                f"Azure OpenAI requires environment variables: {', '.join(missing_vars)}\n"
                "Set them in your .env file or environment:\n"
                "  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/\n"
                "  OPENAI_API_VERSION=2024-02-15-preview\n"
                "  AZURE_OPENAI_API_KEY=your-key (or pass as api_key argument)"
            )
        
        # Also set compatibility variables
        if os.getenv('AZURE_OPENAI_ENDPOINT'):
            os.environ['AZURE_API_BASE'] = os.getenv('AZURE_OPENAI_ENDPOINT')
        if os.getenv('OPENAI_API_VERSION'):
            os.environ['AZURE_API_VERSION'] = os.getenv('OPENAI_API_VERSION')
    
    # Initialize engine using TextGrad's logic
    try:
        llm_engine = tg.get_engine(engine_name=model_string, **engine_kwargs)
        tg.set_backward_engine(llm_engine, override=True)
        
        print(f"[INFO] Initialized engine: {model_string}")
        return llm_engine
        
    except Exception as e:
        # Provide helpful error message
        error_msg = f"Failed to initialize engine '{model_string}': {str(e)}\n\n"
        error_msg += "Common issues:\n"
        error_msg += "  - Missing API key (set via api_key argument or environment variable)\n"
        error_msg += "  - Invalid model string (check TextGrad documentation)\n"
        error_msg += "  - For Azure: missing AZURE_OPENAI_ENDPOINT or OPENAI_API_VERSION\n"
        error_msg += "  - For Ollama: ensure Ollama is running locally\n"
        raise RuntimeError(error_msg) from e

def load_dataset(task_name="DIAGNO3", llm_engine=None):
    """
    Load a dataset for training and testing ACT.
    
    Args:
        task_name: Name of the task to load (default: "BBH_object_counting")
        llm_engine: LLM engine to use for evaluation
    
    Returns:
        tuple: (train_set, val_set, test_set, eval_fn)
    """
    print(f"[INFO] Loading dataset: {task_name}")
    
    train_set, val_set, test_set, eval_fn = load_task(task_name, evaluation_api=llm_engine)
    
    # Convert labels to yes/no format
    train_set = [(x, "yes" if y == 1 else "no") for x, y in train_set]
    val_set = [(x, "yes" if y == 1 else "no") for x, y in val_set]
    test_set = [(x, "yes" if y == 1 else "no") for x, y in test_set]
    
    print(f"[INFO] Dataset loaded - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    return train_set, val_set, test_set, eval_fn

def print_label_distribution(dataset, dataset_name="Dataset"):
    """
    Print the label distribution of a dataset.
    
    Args:
        dataset: List of (input, label) tuples
        dataset_name: Name to display (default: "Dataset")
    
    Returns:
        dict: Label counts
    """
    from collections import Counter
    
    # Extract labels
    labels = [label for _, label in dataset]
    
    # Count occurrences
    label_counts = Counter(labels)
    total = len(labels)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"{dataset_name.upper()} - LABEL DISTRIBUTION")
    print(f"{'='*50}")
    print(f"Total samples: {total}\n")
    
    for label, count in sorted(label_counts.items()):
        percentage = (count / total) * 100
        bar = '█' * int(percentage / 2)  # Visual bar (50 chars = 100%)
        print(f"  {label:>5}: {count:>4} ({percentage:>5.1f}%) {bar}")
    
    print(f"{'='*50}\n")

def train_model(
    train_set,
    llm_engine,
    max_depth=3,
    max_steps_per_node=3,
    num_threads=4,
    max_logical_ops=1,
    max_examples_per_group=20,
    keyword="characteristic",
    data_type="a text",
    min_gini=0.05,
    verbose=True,
):
    """
    Train an ACT model on the provided dataset.
    
    Args:
        train_set: Training data as list of (input, label) tuples
        llm_engine: LLM engine for training
        max_depth: Maximum depth of the decision tree (default: 3)
        max_steps_per_node: Number of optimization steps per node (default: 3)
        num_threads: Number of parallel threads (default: 4)
        max_logical_ops: Max logical operators in questions (default: 1)
        max_examples_per_group: Max examples shown per group during training (default: 20)
        keyword: Focus keyword for LLM (default: "characteristic")
        data_type: Description of data type (default: "a text")
        verbose: Print training progress (default: True)
    
    Returns:
        Trained CARTAgent model
    """
    if verbose:
        print(f"[INFO] Training ACT model with max_depth={max_depth}, max_steps_per_node={max_steps_per_node}")
    
    # Initialize model
    model = fn.CARTAgent(llm_engine, max_depth=max_depth)
    
    # Extract data and labels
    train_data = [x for x, _ in train_set]
    train_labels = [y for _, y in train_set]
    
    # Train the model
    model.fit(
        train_data=train_data,
        train_labels=train_labels,
        llm_engine=llm_engine,
        max_steps_per_node=max_steps_per_node,
        max_depth=max_depth,
        path='root',
        max_workers=num_threads,
        max_logical_ops=max_logical_ops,
        max_examples_per_group=max_examples_per_group,
        keyword=keyword,
        data_type=data_type,
        min_gini=min_gini,
    )
    
    if verbose:
        print(f"[INFO] Model training complete!")
    
    return model


def evaluate_model(
    dataset,
    model,
    eval_fn,
    num_threads=4,
    keyword="characteristic",
    data_type="a text",
    max_samples: int=None,
    verbose=True
):
    """
    Evaluate ACT model on a dataset.
    
    Args:
        dataset: Dataset to evaluate on (list of (input, label) tuples)
        model: Trained CARTAgent model
        eval_fn: Evaluation function from load_task
        num_threads: Number of parallel threads (default: 4)
        keyword: Focus keyword for LLM (default: "characteristic")
        data_type: Description of data type (default: "a text")
        verbose: Print evaluation progress (default: True)
    
    Returns:
        float: Accuracy score (0-1)
    """
    if verbose:
        print(f"[INFO] Evaluating model on {len(dataset)} samples...")
    
    accuracy_list = fn.eval_dataset(
        test_set=dataset,
        eval_fn=eval_fn,
        model=model,
        keyword=keyword,
        data_type=data_type,
        max_workers=num_threads,
        max_samples=max_samples,
    )
    
    accuracy = np.mean(accuracy_list)
    
    if verbose:
        print(f"[INFO] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

def plot_tree(
    model,
    train_set=None,
    llm_engine=None,
    num_threads=4,
    keyword="characteristic",
    data_type="a text",
    save_path=None
):
    """
    Visualize the trained ACT decision tree (simplified version with trimming).
    Trims unnecessary nodes: if all leaves under a node have the same label,
    that node becomes a leaf with that label.
    
    Args:
        model: Trained CARTAgent model
        train_set: Training data (not used in basic version, kept for compatibility)
        llm_engine: LLM engine (not used in basic version, kept for compatibility)
        num_threads: Number of parallel threads (not used in basic version)
        keyword: Focus keyword (not used in basic version)
        data_type: Description of data type (not used in basic version)
        save_path: Optional path to save the plot (e.g., "tree.png")
    
    Returns:
        NetworkX graph representing the tree
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import textwrap
    
    print("[INFO] Generating simplified tree visualization with trimming...")
    
    def get_all_leaf_labels(node):
        """
        Recursively get all leaf labels under a node.
        Returns a set of unique leaf labels.
        """
        if node.is_leaf:
            return {node.label}
        
        labels = set()
        if node.left:
            labels.update(get_all_leaf_labels(node.left))
        if node.right:
            labels.update(get_all_leaf_labels(node.right))
        
        return labels
    
    def trim_tree(node):
        """
        Recursively trim the tree. If all leaves under a node have the same label,
        convert that node to a leaf with that label.
        """
        if node.is_leaf:
            return node
        
        # First, recursively trim children
        if node.left:
            node.left = trim_tree(node.left)
        if node.right:
            node.right = trim_tree(node.right)
        
        # Get all leaf labels under this node
        leaf_labels = get_all_leaf_labels(node)
        
        # If all leaves have the same label, convert this node to a leaf
        if len(leaf_labels) == 1:
            single_label = list(leaf_labels)[0]
            print(f"[TRIM] Converting node with question to leaf (label: {single_label})")
            node.is_leaf = True
            node.label = single_label
            node.left = None
            node.right = None
        
        return node
    
    # Create a deep copy of the tree structure to avoid modifying the original
    import copy
    trimmed_root = copy.deepcopy(model.root)
    
    # Trim the tree
    trimmed_root = trim_tree(trimmed_root)
    
    # Now build the graph from the trimmed tree
    graph = nx.DiGraph()
    node_counter = [0]
    
    def add_node_to_graph(node, parent_id=None, edge_label=""):
        """Recursively add nodes to the graph."""
        current_id = node_counter[0]
        node_counter[0] += 1
        
        if node.is_leaf:
            # Leaf node: show only label
            node_label = f"Predict: {node.label}"
            color = "lightgreen" if node.label == "yes" else "lightcoral"
        else:
            # Internal node: show question
            wrapped_prompt = "\n".join(textwrap.wrap(node.prompt, width=30))
            node_label = wrapped_prompt
            color = "lightblue"
        
        # Add node to graph
        graph.add_node(current_id, label=node_label, color=color, is_leaf=node.is_leaf)
        
        # Add edge from parent
        if parent_id is not None:
            graph.add_edge(parent_id, current_id, label=edge_label)
        
        # Process children
        if not node.is_leaf and node.left is not None and node.right is not None:
            add_node_to_graph(node.left, current_id, "Yes")
            add_node_to_graph(node.right, current_id, "No")
        
        return current_id
    
    # Build the graph from trimmed tree
    add_node_to_graph(trimmed_root)
    
    # Get node attributes
    node_labels = nx.get_node_attributes(graph, 'label')
    node_colors = [graph.nodes[node]['color'] for node in graph.nodes()]
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    try:
        # Try graphviz layout
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    except:
        # Fallback to spring layout
        print("[WARNING] Graphviz not available, using spring layout")
        pos = nx.spring_layout(graph, k=3, iterations=50)
    
    # Draw the tree
    nx.draw(graph, pos,
            with_labels=False,
            node_size=3000,
            node_color=node_colors,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->')
    
    # Draw node labels
    nx.draw_networkx_labels(graph, pos, node_labels, font_size=8, font_color="black")
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=9)
    
    # Title
    total_nodes = len(graph.nodes())
    leaf_nodes = sum(1 for _, data in graph.nodes(data=True) if data['is_leaf'])
    plt.title(f"ACT Decision Tree (Trimmed)\nTotal Nodes: {total_nodes} | Leaf Nodes: {leaf_nodes}", 
              fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        print(f"[INFO] Tree saved to {save_path}")
    
    plt.show()
    
    print(f"✅ Tree visualization complete! (Trimmed to {total_nodes} nodes)")
    return graph

# Example usage and quick start guide
def print_quick_start():
    """Print a quick start guide for the demo."""
    guide = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              ACT Framework - Quick Start Guide               ║
    ╚══════════════════════════════════════════════════════════════╝
    
    1. Setup Engine:
       # OpenAI
       llm_engine = setup_engine("gpt-4.1-nano", api_key="sk-...")
       
       # Ollama (local)
       llm_engine = setup_engine("ollama-gemma3:4b")
    
    2. Load Dataset:
       train, val, test, eval_fn = load_dataset("BBH_object_counting", llm_engine)
    
    3. Train Model:
       model = train_model(train, llm_engine, max_depth=2, max_steps_per_node=3)
    
    4. Evaluate Model:
       accuracy = evaluate_model(test, model, eval_fn)
    
    5. Visualize Tree:
       plot_tree(model, train, llm_engine, save_path="tree.png")
    """
    print(guide)


if __name__ == "__main__":
    print_quick_start()