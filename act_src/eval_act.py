"""
eval_acart.py
=============
Evaluate a saved ACART model's accuracy on train and test sets.

Usage:
    python eval_acart.py --model_path /path/to/model.pkl
"""

import os
import argparse
import pickle
import numpy as np
import textgrad as tg
from textgrad.tasks import load_task
import functions as fn
import act_helper as ah
import time
from dotenv import load_dotenv
load_dotenv(override=True)

def parse_model_path(model_path):
    """Extract task and model name from filename."""
    filename = os.path.basename(model_path)
    parts = filename.replace('.pkl', '').split('_')
    
    task = parts[0]
    
    # Find where depth starts to separate model name
    depth_idx = next(i for i, p in enumerate(parts) if p.startswith('depth'))
    model_parts = parts[1:depth_idx]
    model_name = '_'.join(model_parts)
    print(f"[INFO] Model name detected: {model_name}")
    
    return task, model_name

def get_engine_from_model_name(model_name):
    """Determine engine based on model name."""
    if 'gpt' in model_name.lower():
        # Fix the specific case of gpt_4_1_nano -> gpt-4.1-nano
        if 'nano' in model_name:
            return "azure-gpt-4.1-nano"
        elif 'mini' in model_name:
            return "azure-gpt-4.1-mini"
    elif 'gemma' in model_name.lower():
        return "openai-google/gemma-3-4b-it"
    elif 'qwen' in model_name.lower():
        return "openai-Qwen/Qwen3-4B"
    else:
        raise ValueError(f"Model {model_name} not supported yet")

def _get_all_leaf_labels(node):
    """
    Recursively collect the labels of all leaves below `node`.
    """
    if node is None:
        return set()
    if getattr(node, "is_leaf", False):
        return {node.label}
    
    labels = set()
    if getattr(node, "left", None) is not None:
        labels |= _get_all_leaf_labels(node.left)
    if getattr(node, "right", None) is not None:
        labels |= _get_all_leaf_labels(node.right)
    return labels

def _trim_tree_in_place(node):
    """
    Recursively trim the tree in-place.
    If all leaves under a node share the same label, collapse the subtree
    into a single leaf node with that label.
    """
    if node is None or getattr(node, "is_leaf", False):
        return node

    # First trim children
    if getattr(node, "left", None) is not None:
        node.left = _trim_tree_in_place(node.left)
    if getattr(node, "right", None) is not None:
        node.right = _trim_tree_in_place(node.right)

    # Then check if this node's subtree is homogeneous
    leaf_labels = _get_all_leaf_labels(node)
    if len(leaf_labels) == 1:
        single_label = next(iter(leaf_labels))
        print(f"[TRIM] Converting internal node to leaf (label: {single_label})")
        node.is_leaf = True
        node.label = single_label
        node.left = None
        node.right = None

    return node

def prune_model_tree(model):
    """
    Prune the ACT tree stored in `model.root` by collapsing homogeneous subtrees.
    Modifies `model` in-place.
    """
    if not hasattr(model, "root"):
        raise AttributeError("Model has no `root` attribute; cannot prune tree.")

    print("[INFO] Pruning decision tree (collapsing homogeneous subtrees)...")
    _trim_tree_in_place(model.root)
    print("[INFO] Pruning finished.")

def compute_average_question_length(root):
    """
    Compute the average length of all internal-node questions in the tree.

    Length is measured in characters; if you prefer words, see the commented line.
    """
    lengths_chars = []
    lengths_words = []

    def _dfs(node):
        if node is None:
            return
        # Only internal nodes have questions
        if not getattr(node, "is_leaf", False):
            prompt = getattr(node, "prompt", None)
            if isinstance(prompt, str):
                lengths_chars.append(len(prompt))
                lengths_words.append(len(prompt.split()))
        _dfs(getattr(node, "left", None))
        _dfs(getattr(node, "right", None))

    _dfs(root)

    if not lengths_chars:
        return 0.0, 0.0

    avg_chars = float(np.mean(lengths_chars))
    avg_words = float(np.mean(lengths_words))
    return avg_chars, avg_words

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved ACART model.")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the saved .pkl model file")
    parser.add_argument("--num_threads", type=int, default=16,
                        help="Number of threads for parallel evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--do_full_test_set", action="store_true",
                        help="Add flag to evaluate on whole test set")
    args = parser.parse_args()
    
    # Parse model path
    task, model_name = parse_model_path(args.model_path)
    print(f"[INFO] Task: {task}, Model: {model_name}")
    
    # Determine engine
    engine_name = get_engine_from_model_name(model_name)
    print(f"[INFO] Using engine: {engine_name}")
    
    # Load engine
    llm_api_eval = tg.get_engine(engine_name=engine_name)
    
    # Load model
    print(f"[INFO] Loading model from: {args.model_path}")
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
    model.llm_engine = llm_api_eval

    # Prune model for evaluation
    prune_model_tree(model)

    # Question-length statistics on pruned tree
    avg_q_chars, avg_q_words = compute_average_question_length(model.root)
    print(f"[INFO] Average question length in tree: {avg_q_chars:.1f} characters (~{avg_q_words:.1f} words)")

    # Initialize path-length logging attributes (used in functions.eval_sample)
    model._collect_path_lengths = False
    model._path_lengths = []

    
    # Load dataset and evaluation function
    task_result = load_task(task, evaluation_api=llm_api_eval, seed=args.seed)


    # Handle both 4-item and 5-item returns (for backward compatibility)
    if len(task_result) == 5:
        train_set, val_set, test_set, full_test_set, eval_fn = task_result
        has_full_test = True
    else:
        train_set, val_set, test_set, eval_fn = task_result
        full_test_set = None
        has_full_test = False

    # Print sample entries from datasets
    print("\n" + "="*80)
    print("SAMPLE ENTRIES FROM DATASETS")
    print("="*80)

    print("\n[TRAIN SET SAMPLES]")
    for i in range(min(3, len(train_set))):
        x, y = train_set[i]
        print(f"\nSample {i+1}:")
        print(f"Input (first 200 chars): {x[:200]}...")
        print(f"Label: {y}")

    print("\n[TEST SET SAMPLES]")
    for i in range(min(3, len(test_set))):
        x, y = test_set[i]
        print(f"\nSample {i+1}:")
        print(f"Input (first 200 chars): {x[:200]}...")
        print(f"Label: {y}")


    # Print out information about the data
    print(f"[INFO] Train/Val/Test Set Lengths: {len(train_set)}, {len(val_set)}, {len(test_set)}")
    print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
    print(f"Train data distirbution: label 1: {sum(1 if y == 1 else 0 for x, y in train_set)} label 0: {sum(1 if y == 0 else 0 for x, y in train_set)}")
    print(f"Test data distirbution: label 1: {sum(1 if y == 1 else 0 for x, y in test_set)} label 0: {sum(1 if y == 0 else 0 for x, y in test_set)}")
    if has_full_test:
        print(f"Full test data distirbution: label 1: {sum(1 if y == 1 else 0 for x, y in full_test_set)} label 0: {sum(1 if y == 0 else 0 for x, y in full_test_set)}")

    
    # Convert labels to yes/no format
    train_set = [(x, "yes" if y == 1 else "no") for x, y in train_set]
    test_set = [(x, "yes" if y == 1 else "no") for x, y in test_set]
    if has_full_test:
        full_test_set = [(x, "yes" if y == 1 else "no") for x, y in full_test_set]
    
    # Evaluate model
    print("\n" + "="*50)
    print("EVALUATING MODEL ACCURACY")
    print("="*50)

    keyword, data_type = ah.resolve_data_set_specifics(task=task, keyword='characteristic', data_type='a text')
    print(f"[INFO] Using keyword {keyword} and data type: {data_type}")
    print("[INFO] Evaluating on training set...")
    train_acc, train_fallbacks, train_refusals = fn.eval_dataset(train_set, eval_fn, model, max_workers=args.num_threads, keyword=keyword, data_type=data_type)
    print(f"[INFO] Final training accuracy: {np.mean(train_acc):.4f} with {train_fallbacks} fallbacks returned and {train_refusals} refusals")
    
    try:
        tk_pro_before, tk_pmt_before, tk_com_before = llm_api_eval.total_tokens, llm_api_eval.prompt_tokens, llm_api_eval.completion_tokens
        print(f"[INFO] Token count before eval: {tk_pro_before}, {tk_pmt_before}, {tk_com_before}")
    except:
        print(f"[WARNING] Failed to extract the number of tokens processed during evaluation on the train set")

    print("[INFO] Evaluating on test set...")
    
    # Enable path-length collection for TEST set
    model._collect_path_lengths = True
    model._path_lengths = []

    eval_start_time = time.time()
    test_acc, test_fallbacks, test_refusals = fn.eval_dataset(test_set, eval_fn, model, max_workers=args.num_threads, keyword=keyword, data_type=data_type)
    eval_stop_time = time.time()
    eval_time = eval_stop_time - eval_start_time

    # Stop collecting paths to avoid affecting any later calls
    model._collect_path_lengths = False

    print(f"[INFO] Final test accuracy: {np.mean(test_acc):.4f} with {test_fallbacks} fallbacks returned and {test_refusals} refusals")
    print(f"[INFO] Eval time: {eval_time:.2f} seconds ({eval_time/60:.2f} minutes)")

    # Compute average decision path length on TEST set
    path_lengths = [pl for pl in getattr(model, "_path_lengths", []) if pl is not None]
    if len(path_lengths) > 0:
        avg_test_path_len = float(np.mean(path_lengths))
        print(f"[INFO] Average decision path length on TEST set: {avg_test_path_len:.2f} questions")
    else:
        print("[WARNING] No path-length information collected for TEST set.")

    try:
        tk_pro_after, tk_pmt_after, tk_com_after = llm_api_eval.total_tokens, llm_api_eval.prompt_tokens, llm_api_eval.completion_tokens
        print(f"[INFO] Token count for evaluation on test set -> Total: {tk_pro_after-tk_pro_before}, Prompt: {tk_pmt_after-tk_pmt_before}, Completion: {tk_com_after-tk_com_before}")
    except:
        print(f"[WARNING] Failed to extract the number of tokens processed during evaluation on the test set")


    # Collect correctness + labels, then build confusion & metrics
    test_correct, test_y_true = fn.eval_dataset_with_labels(test_set, eval_fn, model, num_threads=args.num_threads, keyword=keyword, data_type=data_type)
    test_y_pred = fn.reconstruct_predictions_from_correctness(test_y_true, test_correct)
    metrics = fn.compute_binary_metrics(test_y_true, test_y_pred)

    if metrics is None:
        print("[WARNING] Could not compute metrics (labels not mapped to 0/1). See Step 7.")
    else:
        print("[INFO] Confusion matrix (tn, fp, fn, tp):", metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"])
        print(f"[INFO] Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | MCC: {metrics['mcc']:.4f}")

    # Evaluate on FULL test set if available
    if has_full_test and args.do_full_test_set:
        print(f"\n[INFO] Evaluating on FULL UNBALANCED test set (n={len(full_test_set)})...")
        full_test_acc, full_test_fallbacks, full_test_refusals = fn.eval_dataset(full_test_set, eval_fn, model, max_workers=args.num_threads, keyword=keyword, data_type=data_type)
        print(f"[INFO] Final full test accuracy: {np.mean(full_test_acc):.4f} with {full_test_fallbacks} fallbacks returned and {full_test_refusals} refusals")
        full_test_correct, full_test_y_true = fn.eval_dataset_with_labels(full_test_set, eval_fn, model, num_threads=args.num_threads, keyword=keyword, data_type=data_type)
        full_test_y_pred = fn.reconstruct_predictions_from_correctness(full_test_y_true, full_test_correct)
        full_test_metrics = fn.compute_binary_metrics(full_test_y_true, full_test_y_pred)
        print(f"TEST (Full) Precision: {full_test_metrics['precision']:.3f}, Recall: {full_test_metrics['recall']:.3f}, F1: {full_test_metrics['f1']:.3f}, MCC: {full_test_metrics['mcc']:.3f}")
        print(f"TEST (Full) Confusion Matrix - TN: {full_test_metrics['tn']}, FP: {full_test_metrics['fp']}, FN: {full_test_metrics['fn']}, TP: {full_test_metrics['tp']}")

    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Task: {task}")
    print(f"Engine: {engine_name}")
    print(f"Training Accuracy: {np.mean(train_acc):.4f}")
    print(f"Test Accuracy: {np.mean(test_acc):.4f}")
    print(f"Generalization Gap: {np.mean(train_acc) - np.mean(test_acc):.4f}")

if __name__ == "__main__":
    main()