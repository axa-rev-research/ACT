"""
act_helper.py
============

Contains helper functions for the Agentic Classification Tree (ACT) implementation.

This module is imported by several scripts in act_src/ and is not meant to be run directly.

Code has 4 Parts:
- Part 1 - Helper functions for logging and outputs
- Part 2 - Helper functions for dataset specific settings
- Part 3 - Helper functions for checkpointing
- Part 4 - Helper functions for core implementation of act (act_functions.py)
"""

import os
import csv
import numpy as np
import random
import textwrap
import tiktoken
import warnings
import pickle
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning, module="httpx")
from collections import Counter
from typing import Sequence, Tuple, Any, Union


# Part 1 - Helper functions for logging and outputs

def log_training_stats(csv_path, stats_dict):
    """
    Log training statistics to a CSV file.
    Creates the file with headers if it doesn't exist, otherwise appends.
    """
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats_dict.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(stats_dict)

def check_output_path(output_path):
    """Check if output path exists, raise exception if not."""
    path = Path(output_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Output path does not exist: {output_path}")
    
    if not path.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_path}")
    
    # Optional: Check if path is writable
    if not path.stat().st_mode & 0o200:  # Check write permission
        raise PermissionError(f"Output path is not writable: {output_path}")

def print_label_distribution_and_warn(
    train_set: Sequence[Tuple[Any, Union[str, int]]],
    test_set: Sequence[Tuple[Any, Union[str, int]]],
    yes_label: str = "yes",
    threshold_pp: float = 1.5,
    name_train: str = "train",
    name_test: str = "test",
) -> None:
    """
    Computes and prints the fraction of positive/yes labels in train and test.
    Warns if the absolute difference exceeds `threshold_pp` percentage points.

    Accepts labels as:
      - strings: "yes"/"no" (or custom `yes_label`)
      - ints: 1/0
    """

    def is_yes(y) -> bool:
        if isinstance(y, str):
            return y.strip().lower() == yes_label.strip().lower()
        if isinstance(y, (int, bool)):
            return int(y) == 1
        raise TypeError(f"Unsupported label type: {type(y)} ({y!r})")

    def summarize(name: str, ds: Sequence[Tuple[Any, Union[str, int]]]) -> Tuple[int, int, float]:
        n = len(ds)
        yes = sum(1 for _, y in ds if is_yes(y))
        frac = yes / n if n else 0.0
        print(f"[INFO] {name}: yes={yes}/{n} ({frac*100:.2f}%)")
        return yes, n, frac

    _, _, frac_train = summarize(name_train, train_set)
    _, _, frac_test = summarize(name_test, test_set)

    diff_pp = abs(frac_train - frac_test) * 100.0
    if diff_pp > threshold_pp:
        warnings.warn(
            f"Label distribution shift: |{name_train}-{name_test}| = {diff_pp:.2f}pp "
            f"(threshold {threshold_pp:.2f}pp).",
            RuntimeWarning,
        )

# Part 2 - Helper functions for dataset specific settings
    
def resolve_data_set_specifics(task, keyword, data_type):
    if keyword != 'characteristic' or data_type != 'a text':
        print(f"[WARNING] using self set keywords and data type")
        return keyword, data_type
    else:
        if task == "DIAGNO3" or task == "DIAGNO_FULL":
            return 'symptom', "a patient's symptom description"
        elif task == "SPAM" or task == "SPAM_FULL_DATASET":
            return 'characteristic', "an E-MAIL"
        elif task == "JAILBREAK":
            return 'characteristic', "an LLM prompt"
        elif task == "BANKCHURN" or task == "BANKCHURN_IMBALANCED" or task == "BANKCHURN_FULL":
            return 'characteristic', "a bank customer profile"
        elif task == "IMDBFULL" or task == "IMDBBALANCED":
            return 'characteristic', "a movie review"
        else:
            print(f"[WARNING] task '{task}' is NEW")
            return keyword, data_type

# Part 3 - Helper functions for checkpointing

def cleanup_checkpoints(checkpoint_dir, train_run_name):
    """
    Delete all checkpoint files for a specific task after successful training.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        train_run_name: Name identifier for the train run (used in checkpoint filenames)
    """
    if not os.path.exists(checkpoint_dir):
        print(f"[CHECKPOINT] No checkpoint directory found at {checkpoint_dir}")
        return
    
    deleted_count = 0
    for filename in os.listdir(checkpoint_dir):
        if train_run_name in filename and filename.endswith('.pkl'):
            filepath = os.path.join(checkpoint_dir, filename)
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"[WARNING] Could not delete checkpoint {filepath}: {e}")
    
    print(f"[CHECKPOINT] Cleaned up {deleted_count} checkpoint file(s) for task '{train_run_name}'")

def save_checkpoint(node, checkpoint_path, step_info):
    """
    Save a checkpoint of the current training state.
    
    Args:
        node: The current CARTNode being trained
        checkpoint_path: Path to save the checkpoint
        step_info: Dictionary containing training progress information
    """
    checkpoint = {
        'node_state': {
            'depth': node.depth,
            'is_leaf': node.is_leaf,
            'label': node.label,
        },
        'step_info': step_info,
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"[CHECKPOINT] Saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path):
    """Load a training checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"[CHECKPOINT] Loaded from {checkpoint_path}")
    return checkpoint

def get_checkpoint_path(output_directory, train_run_name, path, step):
    """Generate a consistent checkpoint path."""
    safe_path = path.replace('/', '_')
    return os.path.join(output_directory, f"{train_run_name}_{safe_path}_step{step}.pkl")

# Part 4 - Helper functions for core implementation of act (act_functions.py)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def gini_impurity(labels):
    '''Returns gini impurity of a list of class labels'''
    if not isinstance(labels, list):
        raise ValueError(f"Expected a list for labels, but got {type(labels)}")
    if len(labels) == 0:
        return 0.0
    total = len(labels)
    counts = Counter(labels)
    return 1 - sum((count / total) ** 2 for count in counts.values())

def format_distribution(labels):
    '''Returns distribution (label: number of occurences) for a list of class labels'''
    if not isinstance(labels, list):
        raise ValueError(f"Expected a list for labels, but got {type(labels)}")
    counts = Counter(labels)
    return ", ".join(f"{k}:{v}" for k, v in counts.items())

def majority(labels):
    print("[INFO] label yes count: ", labels.count("yes"), " and label no count: ", labels.count("no"))
    return "yes" if labels.count("yes") >= labels.count("no") else "no"

def wrap_label(text, width=40):
    return "\n".join(textwrap.wrap(text, width)) if text else ""

def stopping_criteria(labels, min_samples, max_depth, depth, min_gini):
    '''Returns boolean to indicate if exploring from the current node should continue or not.
    If exploring stops the function also returns the reason why.'''

    if not isinstance(labels, list):
        raise TypeError(f"Expected list for labels, got {type(labels).__name__}")

    # Stop if pure, too small, or max depth reached
    gini = gini_impurity(labels=labels)
    if len(set(labels)) == 1 or len(labels) < min_samples or (max_depth is not None and depth >= max_depth) or (gini < min_gini):
        reason = ""
        if len(set(labels)) == 1:
            reason = f"group only has {labels[:1]} labels"
        elif len(labels) < min_samples:
            reason = f"group has less than {min_samples} samples"
        elif max_depth is not None and depth >= max_depth:
            reason = f"reached max depth of {max_depth}"
        elif gini < 0.1:
            reason = f"has a gini of <= {min_gini}"
        else:
            raise Exception(f"Stopping criteria triggered with no valid reason")
        return True, reason
    else:
        return False, ""

def count_tokens(text, model_name="gpt-4"):
    """Count tokens using tiktoken with a safe fallback."""
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(str(text)))