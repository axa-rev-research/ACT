"""
vis_act.py
===========
Visualize a saved ACT model by extracting metadata from filename.

Usage:
    python vis_tree.py --model_path /path/to/model.pkl
"""

import os
import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from dotenv import load_dotenv
load_dotenv(override=True)

import textgrad as tg
from textgrad.tasks import load_task

import functions as fn
import act_helper as ah
import function_prompts as fp
import random


# Part 6 - Visualize ACART tree

def plot_acart_tree(root, data, labels, llm_engine, max_workers, keyword, data_type):
    """
    Create a NetworkX graph representation of the ACART tree with training statistics.
    
    Args:
        root: The root CARTNode of the trained tree
        data: Training data used to fit the tree
        labels: Training labels corresponding to the data
        llm_engine: LLM engine used for inference
    
    Returns:
        NetworkX DiGraph representing the tree structure
    """

    graph = nx.DiGraph()
    node_counter = [0]  # Use list to maintain reference in nested function
    
    def add_node_to_graph(node, data_subset, labels_subset, parent_id=None, edge_label=""):
        """Recursively add nodes to the graph with their training statistics."""
        current_id = node_counter[0]
        node_counter[0] += 1
        
        # Calculate node statistics
        total_samples = len(labels_subset)
        label_counts = Counter(labels_subset)
        majority_label = max(label_counts, key=label_counts.get) if label_counts else "unknown"
        accuracy = label_counts.get(majority_label, 0) / total_samples if total_samples > 0 else 0.0
        gini = ah.gini_impurity(labels_subset) if labels_subset else 0.0
        
        # Create node label with statistics
        if node.is_leaf:
            # For leaf nodes, show prediction, sample counts, and accuracy
            node_label = (f"LEAF\n"
                         f"Predicts: {node.label}\n"
                         f"Samples: {total_samples}\n"
                         f"Distribution: {dict(label_counts)}\n"
                         f"Accuracy: {accuracy:.2f}")
            label_value = node.label
        else:
            # For internal nodes, show the splitting question and statistics
            wrapped_prompt = ah.wrap_label(node.prompt, width=30)
            node_label = (f"{wrapped_prompt}\n"
                         f"Samples: {total_samples}\n"
                         f"Gini: {gini:.3f}\n"
                         f"Distribution: {dict(label_counts)}")
            label_value = None
        
        # Add node to graph
        graph.add_node(current_id, 
                      label=node_label,
                      is_leaf=node.is_leaf,
                      label_value=label_value,
                      samples=total_samples,
                      accuracy=accuracy,
                      gini=gini,
                      distribution=dict(label_counts))
        
        # Add edge from parent if exists
        if parent_id is not None:
            graph.add_edge(parent_id, current_id, label=edge_label)
        
        # Process children for internal nodes
        if not node.is_leaf and node.left is not None and node.right is not None:

            # Helper function for infering over the data
            @fn.retry_with_backoff(max_retries=3, initial_delay=0.5)
            def evaluate_with_agent(prompt, context, keyword, data_type):

                system_prompt = fp.get_node_inference_system_prompt(question=prompt, keyword=keyword, data_type=data_type)

                # Initiallize LLM Agent used that classifies the samples using the prompt in the node
                agent_llm = tg.BlackboxLLM(
                    engine=llm_engine,
                    system_prompt=tg.Variable(
                        system_prompt,
                        requires_grad=False,
                        role_description="System prompt for the model to provide a yes/no answer"
                    )
                )

                # Use agent_llm passed as argument, not global
                def _do_call():
                    return agent_llm(tg.Variable(
                        context,
                        requires_grad=False,
                        role_description="prompt that instructs the LLM to answer the question with yes/no based on the context provided.")
                        ).value.lower()
                try:
                    result = fn.call_with_timeout(_do_call, timeout_s=180)
                except Exception as e:
                    print(f"[WARNING] evaluate_with_agent timeout/error")
                    # Fallback: vote "no" to keep control flow moving
                    return 'error'
                
                end_of_response = result[-30:].lower()
                think_end = result.lower().find('</think>')
                start_pos = (think_end + 8) if think_end != -1 else 0
                start_of_response = result[start_pos:start_pos + 30].lower()
                # Returns True if "yes" in result, False otherwise
                if 'yes' in end_of_response:
                    return 'yes'
                elif 'no' in end_of_response:
                    return 'no'
                elif 'yes' in start_of_response:
                    return 'yes'
                elif 'no' in start_of_response:
                    return 'no'
                else:
                    print(f"[WARNING] Could not extract yes or no from answer in evaluate with agent, therefor did not use this example")
                    return 'error'

            left_data, left_labels, right_data, right_labels, error_data, error_labels, _, _, _, _, _ = fn.split_data(max_workers=max_workers,
                                                                                        examples=data_subset,
                                                                                        labels=labels_subset,
                                                                                        split_func=evaluate_with_agent,
                                                                                        prompt_var=node.prompt,
                                                                                        step=current_id,
                                                                                        max_steps_per_node="Plotting",
                                                                                        path="at plotting time",
                                                                                        keyword=keyword,
                                                                                        data_type=data_type)

            
            if isinstance(error_data, list) and isinstance(error_labels, list) and all(item in ['yes', 'no'] for item in error_labels):
                random.shuffle(error_data)
                random.shuffle(error_labels)
            else:
                raise ValueError(f"[ERROR] Error examples and error label list have an unexpected behaviour. They are {type(best_error_examples)} and {type(best_error_labels)}.")
            
            if len(error_data) == len(error_labels):
                mid_point = len(error_data) // 2
            else:
                raise ValueError(f"[ERROR] In add_node_to_graph the length of the error lists are not the same")
            
            left_data.extend(error_data[:mid_point])
            right_data.extend(error_data[mid_point:])
            left_labels.extend(error_labels[:mid_point])
            right_labels.extend(error_labels[mid_point:])

            if len(left_data) != len(left_labels) or len(right_data) != len(right_labels) or not all(item in ['yes', 'no'] for item in left_labels) or not all(item in ['yes', 'no'] for item in right_labels):
                raise ValueError(f"[ERROR] Ina add_node_to_graph the final constructed data subsets have an issue!")

            # Recursively add child nodes
            add_node_to_graph(node.left, left_data, left_labels, current_id, "Yes")
            add_node_to_graph(node.right, right_data, right_labels, current_id, "No")
        
        return current_id
    
    # Start building the graph from root
    add_node_to_graph(root, data, labels)
    return graph

def calculate_final_weighted_gini(graph):
    """
    Calculate weighted Gini for the last layer of non-leaf nodes.
    """
    # Find all non-leaf nodes
    non_leaf_nodes = [node_id for node_id, data in graph.nodes(data=True) if not data['is_leaf']]
    
    # Find nodes whose children are both leaves (last layer of non-leaf nodes)
    last_layer_non_leaf = []
    for node_id in non_leaf_nodes:
        children = list(graph.successors(node_id))
        if len(children) == 2:
            # Check if both children are leaves
            if all(graph.nodes[child]['is_leaf'] for child in children):
                last_layer_non_leaf.append(node_id)
    
    # Calculate weighted Gini
    total_samples = 0
    weighted_gini_sum = 0
    
    for node_id in last_layer_non_leaf:
        node_gini = graph.nodes[node_id]['gini']
        node_samples = graph.nodes[node_id]['samples']
        
        total_samples += node_samples
        weighted_gini_sum += node_gini * node_samples
        
        print(f"Node {node_id}: Gini={node_gini:.3f}, Samples={node_samples}")
    
    if total_samples > 0:
        final_weighted_gini = weighted_gini_sum / total_samples
    else:
        final_weighted_gini = 0.0
    
    return final_weighted_gini, last_layer_non_leaf


def visualize_acart_tree(root, data, labels, llm_engine, max_workers, keyword, data_type, save_path=None):
    """
    Generate and display a plot of the CART decision tree with comprehensive statistics.
    
    Args:
        root: The root CARTNode of the trained tree
        data: Training data used to fit the tree
        labels: Training labels corresponding to the data
        llm_engine: LLM engine used for inference
        save_path: Optional path to save the plot (supports .png and .pdf)
    """
    
    print("[INFO] Generating tree visualization...")
    
    # Create the graph representation
    graph = plot_acart_tree(root, data=data, labels=labels, llm_engine=llm_engine, max_workers=max_workers, keyword=keyword, data_type=data_type)
    node_labels = nx.get_node_attributes(graph, 'label')

    # Calculate final weighted Gini
    final_weighted_gini, last_layer_nodes = calculate_final_weighted_gini(graph)
    print(f"\n[INFO] Final Weighted Gini (last non-leaf layer): {final_weighted_gini:.4f}")
    print(f"[INFO] Last layer non-leaf nodes: {last_layer_nodes}")
    
    # Color-code nodes based on type and prediction
    colors = []
    for node_id in graph.nodes:
        node_data = graph.nodes[node_id]
        is_leaf = node_data['is_leaf']
        
        if is_leaf:
            label_value = node_data['label_value']
            if label_value == "yes":
                colors.append("lightgreen")
            elif label_value == "no":
                colors.append("lightcoral")
            else:
                colors.append("lightgray")
        else:
            # Color internal nodes based on their Gini impurity
            gini = node_data.get('gini', 0.5)
            if gini < 0.2:
                colors.append("lightblue")  # Pure splits
            elif gini < 0.4:
                colors.append("lightyellow")  # Moderate splits
            else:
                colors.append("lightpink")  # Poor splits
    
    # Create the plot
    plt.figure(figsize=(16, 12))
    
    try:
        # Try to use graphviz layout if available
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    except:
        # Fallback to hierarchical layout
        print("[WARNING] Graphviz not available, using spring layout")
        pos = nx.spring_layout(graph, k=3, iterations=50)
    
    # Draw the tree
    nx.draw(graph, pos, 
            with_labels=False, 
            node_size=4000, 
            node_color=colors, 
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->')
    
    # Draw node labels
    nx.draw_networkx_labels(graph, pos, node_labels, font_size=7, font_color="black")
    
    # Draw edge labels (Yes/No)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=8)
    
    # Add title and legend
    plt.title("ACART Decision Tree\n(Green=Yes prediction, Red=No prediction, Blue/Yellow/Pink=Internal nodes by Gini)", 
              fontsize=14, pad=20)
    
    # Add statistics summary
    total_nodes = len(graph.nodes)
    leaf_nodes = sum(1 for _, data in graph.nodes(data=True) if data['is_leaf'])
    internal_nodes = total_nodes - leaf_nodes
    
    stats_text = f"Nodes: {total_nodes} total ({internal_nodes} internal, {leaf_nodes} leaves)"
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot if path is provided
    if save_path:
        if save_path.endswith(".pdf"):
            plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        elif save_path.endswith(".png"):
            plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        else:
            # Default to PNG if no extension specified
            save_path += ".png"
            plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        print(f"[INFO] Decision tree saved to {save_path}")
    
    plt.show()
    
    # Print tree statistics
    print("\n" + "="*50)
    print("TREE STATISTICS SUMMARY")
    print("="*50)
    
    for node_id in graph.nodes:
        node_data = graph.nodes[node_id]
        if node_data['is_leaf']:
            print(f"Leaf {node_id}: Predicts '{node_data['label_value']}' | "
                  f"Samples: {node_data['samples']} | "
                  f"Distribution: {node_data['distribution']} | "
                  f"Accuracy: {node_data['accuracy']:.3f}")
    
    return graph

def parse_model_path(model_path):
    """Extract task, model name, and other info from the filename."""
    filename = os.path.basename(model_path)
    # Example: DIAGNO3_gpt_4_1_nano_depth3_steps20_acc0.625.pkl
    parts = filename.replace('.pkl', '').split('_')
    
    task = parts[0]
    
    # Find where depth starts to separate model name from parameters
    depth_idx = next(i for i, p in enumerate(parts) if p.startswith('depth'))
    model_parts = parts[1:depth_idx]
    model_name = '_'.join(model_parts)
    
    return task, model_name

def get_engine_from_model_name(model_name):
    """Determine engine based on model name."""
    if 'gpt' in model_name.lower() and 'nano' in model_name.lower():
            return "azure/gpt-4.1-nano"
    if 'gpt' in model_name.lower() and 'mini' in model_name.lower():
            return "azure/gpt-4.1-mini"
    if 'gemma' in model_name.lower() and '4b' in model_name.lower():
            return "openai-google/gemma-3-4b-it"
    if 'qwen' in model_name.lower() and '4b' in model_name.lower():
            return "openai-Qwen/Qwen3-4B"
    else:
        raise ValueError(f" [ERROR] The model infered from the .pkl path: {model_name} is either wrong or does not correspond to one of the 4 hard coded version that this code accepts. Add them to get_engine_from_model_name function in vis_act.py")

def main():
    parser = argparse.ArgumentParser(description="Visualize a saved ACART model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved .pkl model file")
    args = parser.parse_args()
    
    # Parse model path to extract information
    task, model_name = parse_model_path(args.model_path)
    print(f"[INFO] Extracted task: {task}, model: {model_name}")

    if task not in ['DIAGNO3', 'SPAM', 'JAILBREAK', 'BANKCHURN', 'IMDBBALANCED', 'IMDBFULL']:
        raise ValueError(f"Task {task} is not supported")
    
    keyword, data_type = ah.resolve_data_set_specifics(task, 'characteristic', 'a text')
    
    # Setup output path (same location and name, just .png instead of .pkl)
    output_path = args.model_path.replace('.pkl', '.png')
    
    # Determine engine
    engine_name = get_engine_from_model_name(model_name)
    print(f"[INFO] Using engine: {engine_name}")
    
    # Load engine
    llm_api_eval = tg.get_engine(engine_name=engine_name)
    
    # Load model
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
    model.llm_engine = llm_api_eval
    print("[INFO] Model loaded successfully")
    
    # Load dataset
    train_set, _, _, _ = load_task(task, evaluation_api=llm_api_eval)
    train_set = [(x, "yes" if y == 1 else "no") for x, y in train_set]
    data = [x for x, _ in train_set]
    labels = [y for _, y in train_set]
    print(f"[INFO] Loaded {len(train_set)} training samples")
    
    # Generate visualization
    print("\n" + "="*50)
    print("GENERATING TREE VISUALIZATION")
    print("="*50)
    
    visualize_acart_tree(
        root=model.root,
        data=data,
        labels=labels,
        llm_engine=llm_api_eval,
        max_workers=8,
        keyword=keyword,
        data_type=data_type,
        save_path=output_path
    )
    
    print(f"[SUCCESS] Visualization saved to: {output_path}")

if __name__ == "__main__":
    main()