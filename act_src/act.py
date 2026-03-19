"""
act.py
==========

Main training and evaluation script for the Agentic Classification Tree (ACT).
Loads a dataset, trains the tree with LLM-based splits, evaluates accuracy, and
saves the model to disk.

Usage:
    python act.py --task <TASK_NAME> --max_depth 3 --max_steps_per_node 10
"""

import os
import argparse
import numpy as np
import random
import time
import csv
from datetime import datetime
import textgrad as tg
from textgrad.tasks import load_task
import pickle
from pathlib import Path
import functions as fn
import function_prompts as fp
import act_helper as ah
import vis_act as va
from dotenv import load_dotenv
load_dotenv(override=True)

# Load arguments
def config():
    parser = argparse.ArgumentParser(description="Create a natural language tree for binary classification.")

    # Task and model configuration
    parser.add_argument("--task", type=str, default="DIAGNO3", help="The task to evaluate the model on.")
    parser.add_argument("--model", type=str, default='azure-gpt-4.1-nano', help="Model string for textgrad (e.g., 'azure-gpt-4.1-nano', 'ollama-gemma3:4b')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Training configuration
    parser.add_argument("--do_two_stage_training", action="store_true", help="Enable two-stage training.")
    parser.add_argument("--optimization_constraints", type=str, default="exploring", choices=["basic", "exploring", "exploiting", "demo_notebook"], help="Wheter to do training in two stages or not")
    parser.add_argument("--train_pct", type=float, default=100.0, help="Percentage of the training set to use (1-100).")
    parser.add_argument("--num_threads", type=int, default=16, help="The number of threads to use for evaluation.")
    parser.add_argument("--keyword", type=str, default='characteristic', help="Keyword on which the LLM should focus")
    parser.add_argument("--data_type", type=str, default='a text', help="Additional information about the data set to hint the LLM in the right direction")

    # Tree parameters
    parser.add_argument("--max_depth", type=int, default=3, help="The maximum depth of the decision tree.")
    parser.add_argument("--max_steps_per_node", type=int, default=3, help="Maximum number of optimization steps per node.")
    parser.add_argument("--max_examples_per_group", type=int, default=50, help="Maximum number of examples to show per group in analyze_group_purity.")
    parser.add_argument("--max_logical_ops", type=int, default=2, help="Maximum number of logical operators ('and'/'or') allowed in a splitting question.")
    parser.add_argument("--stop_min_samples", type=int, default=10, help="Minimum number of samples needed in a node to continue splitting this node")
    parser.add_argument("--min_gini", type=float, default=0.01, help="Minimum gini impurity needed in a node to continue splitting this node")
    
    # Checkpoint configuration
    parser.add_argument("--checkpoint_dir", type=str, default="act_train_chckpts", help="Directory to save checkpoints. If None, no checkpointing.")
    parser.add_argument("--checkpoint_frequency", type=int, default=5, help="Save checkpoint every N optimization steps.")

    # Output configuration
    parser.add_argument("--no_save_model", action="store_true", help="Disable saving the model as .pkl at the end.")
    parser.add_argument("--no_checkpoint_save", action="store_true", help="Disable saving the model checkpoints during training.")
    parser.add_argument("--do_vis", action="store_true", help="Enable visualization of the ACT.")
    parser.add_argument("--eval_full_test_set", action="store_true", help="Add flag to evaluate on whole test set")
    parser.add_argument("--out_dir", type=str, default="ACT/act_output/", help="Output directory for logs, model weights and visualization")

    args = parser.parse_args()
    
    # Validate train_pct
    if not 1.0 <= args.train_pct <= 100.0:
        parser.error("--train_pct must be between 1 and 100.")
    
    # Validate output paths
    ah.check_output_path(args.out_dir)
    ah.check_output_path(args.checkpoint_dir)
    
    return args

def main():

    # Load command line arguments
    args = config()

    # Setup Engine
    ah.set_seed(args.seed)
    print(f"[INFO] Using model: {args.model}")
    clean_model_name = args.model.replace(".", "_").replace("-", "_").replace("/", "_")
    llm_api_eval = tg.get_engine(engine_name=args.model)
    print("[INFO] llm_api_eval:", type(llm_api_eval))
    print("[INFO] llm_api_eval.__dict__:", llm_api_eval.__dict__)
    tg.set_backward_engine(llm_api_eval, override=True)

    # Load the data and the evaluation function
    print("[INFO] Loading data...")
    task_result = load_task(args.task, evaluation_api=llm_api_eval, seed=args.seed)

    # Depending on the task, there are 4 or 5 variables
    if len(task_result) == 5:
        train_set, _, test_set, full_test_set, eval_fn = task_result
        has_full_test = True
    else:
        train_set, _, test_set, eval_fn = task_result
        full_test_set = None
        has_full_test = False

    print(f"[INFO] Train/Test Set Lengths: {len(train_set)}, {len(test_set)}")

    # Convert 1 -> yes and 0 -> no
    train_set = [(x, "yes" if y == 1 else "no") for x, y in train_set]
    test_set = [(x, "yes" if y == 1 else "no") for x, y in test_set]
    if has_full_test and args.eval_full_test_set:
        full_test_set = [(x, "yes" if y == 1 else "no") for x, y in full_test_set]
        print(f"[INFO] Full Test Set Length: {len(full_test_set)}")
    print("[INFO] Converted labels to 'yes'/'no'.")

    # Subsample training set by percentage (deterministic)
    pct = float(args.train_pct)
    if pct < 100.0:
        n_total = len(train_set)
        n_keep = max(1, int(round(n_total * pct / 100.0)))

        if n_keep < n_total:
            # Use a local RNG to keep determinism without affecting global state
            rng = random.Random(args.seed)
            train_set = rng.sample(train_set, n_keep)
        print(f"[INFO] Subsampled training set: {n_keep}/{n_total} ({pct:.1f}%)")

    # Compare train/test distribution 
    ah.print_label_distribution_and_warn(train_set, test_set, threshold_pp=1.5)

    # Load train data
    train_data = [x for x, _ in train_set]
    train_labels = [y for _, y in train_set]

    print(f"[INFO] Data loading for {args.task} successful!")

    # Initialize model
    print("[INFO] Initializing model...")
    model = fn.CARTAgent(llm_api_eval, max_depth=args.max_depth)

    # Set task specific variables and print out information about training run configuration
    print(f"[INFO] Saving checkpoints every {args.checkpoint_frequency} steps to {args.checkpoint_dir}")
    keyword, data_type = ah.resolve_data_set_specifics(task=args.task, keyword=args.keyword, data_type=args.data_type)
    print(f"[INFO] Using keyword '{keyword}' and data type '{data_type}'")
    # Set training stages and constraints
    print(f'''[INFO] Start training using {args.num_threads} threads,
          two training stage = {args.do_two_stage_training},
          constraints = {args.optimization_constraints}''')
    constraints = fp.get_prompt_constraints(keyword=keyword,
                                            max_logical_ops=args.max_logical_ops,
                                            training_stage=args.optimization_constraints)
    print(f"[INFO] the prompt constraints are: {constraints}")
    print("[INFO] Model successfully initialized!")
    print("[INFO] Start training...")

    # Keep track of run
    start_time = time.time()
    train_run_name = f"{args.task}_{clean_model_name}_{args.max_depth}_{args.max_steps_per_node}"
    
    # Fit model
    model.fit(
        train_data=train_data,
        train_labels=train_labels,
        llm_engine=llm_api_eval,
        max_steps_per_node=args.max_steps_per_node,
        max_depth=args.max_depth,
        path='root',
        max_workers=args.num_threads,
        max_logical_ops=args.max_logical_ops,
        max_examples_per_group=args.max_examples_per_group,
        keyword=keyword,
        data_type=data_type,
        two_stage_training=args.do_two_stage_training,
        start_opt_constraints=args.optimization_constraints,
        stop_min_samples=args.stop_min_samples,
        min_gini=args.min_gini,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_frequency=args.checkpoint_frequency,
        train_run_name=train_run_name,
        save_checkpoint= not args.no_checkpoint_save,
    )

    # Store metadata about training
    end_time = time.time()
    training_time = end_time - start_time
    print("[INFO] Training terminated successfully!")
    print(f"[INFO] Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    total_tokens_processed, prompt_tokens, completion_tokens = None, None, None
    try:
        print(f"[INFO] Token statistics for ACT training")
        print(f"[INFO] Total tokens processed: {llm_api_eval.total_tokens:,}")
        print(f"[INFO] Prompt tokens: {llm_api_eval.prompt_tokens:,}")
        print(f"[INFO] Completion tokens: {llm_api_eval.completion_tokens:,}")
        total_tokens_processed = llm_api_eval.total_tokens
        prompt_tokens = llm_api_eval.prompt_tokens
        completion_tokens = llm_api_eval.completion_tokens
    except:
        print(f"[WARNING] Failed to extract the number of tokens processed during training")

    # Evaluate model
    train_acc, train_fallbacks, train_refused = fn.eval_dataset(train_set, eval_fn, model, max_workers=args.num_threads, keyword=keyword, data_type=data_type)
    print(f"[INFO] Final training accuracy: {np.mean(train_acc)} with {train_fallbacks} fallback labels returned and {train_refused} refusals")
    
    try:
        tk_pro_before, tk_pmt_before, tk_com_before = llm_api_eval.total_tokens, llm_api_eval.prompt_tokens, llm_api_eval.completion_tokens
    except:
        print(f"[WARNING] Failed to extract the number of tokens processed during evaluation on the train set")

    # Log metadata about evaluation on test set    
    start_eval_time = time.time()
    test_acc, test_fallbacks, test_refused = fn.eval_dataset(test_set, eval_fn, model, max_workers=args.num_threads, keyword=keyword, data_type=data_type)
    end_eval_time = time.time()
    eval_time = end_eval_time - start_eval_time
    print(f"[INFO] Final test accuracy: {np.mean(test_acc)} with {test_fallbacks} fallback labels returned and {test_refused} refusals")
    print(f"[INFO] Eval time: {eval_time:.2f} seconds ({eval_time/60:.2f} minutes)")
    try:
        tk_pro_after, tk_pmt_after, tk_com_after = llm_api_eval.total_tokens, llm_api_eval.prompt_tokens, llm_api_eval.completion_tokens
        print(f"[INFO] Token count for evaluation on test set -> Total: {tk_pro_after-tk_pro_before}, Prompt: {tk_pmt_after-tk_pmt_before}, Completion: {tk_com_after-tk_com_before}")
    except:
        print(f"[WARNING] Failed to extract the number of tokens processed during evaluation on the test set")

    # Additional stats
    test_correct, test_y_true = fn.eval_dataset_with_labels(test_set, eval_fn, model, num_threads=args.num_threads, keyword=keyword, data_type=data_type)
    test_y_pred = fn.reconstruct_predictions_from_correctness(test_y_true, test_correct)
    metrics = fn.compute_binary_metrics(test_y_true, test_y_pred)

    if metrics is None:
        print("[WARNING] Could not compute metrics")
    else:
        print("[INFO] Confusion matrix (tn, fp, fn, tp):", metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"])
        print(f"[INFO] Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | MCC: {metrics['mcc']:.4f}")


    # Log training stats
    # Prepare statistics dictionary
    try:
        stats = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'task': args.task,
            'model': args.model,
            'seed': args.seed,
            'max_depth': args.max_depth,
            'max_steps_per_node': args.max_steps_per_node,
            'train_pct': args.train_pct,
            'num_threads': args.num_threads,
            'max_logical_ops': args.max_logical_ops,
            'stop_min_samples': args.stop_min_samples,
            'min_gini': args.min_gini,
            'train_samples': len(train_set),
            'test_samples': len(test_set),
            'train_accuracy': np.mean(train_acc),
            'test_accuracy': np.mean(test_acc),
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'total_tokens_processed': total_tokens_processed,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'two_stage_training': args.do_two_stage_training,
            'optimization_constraints': args.optimization_constraints,
        }
        # Log to CSV
        csv_path = os.path.join(args.out_dir, "CM_training_logs.csv")
        ah.log_training_stats(csv_path, stats)
        print(f"[INFO] Training statistics logged to {csv_path}")
    except:
        print("[WARNING] Could not save training stats!")

    # Save model
    if not args.no_save_model:

        # Set model name
        save_model_path = os.path.join(args.out_dir, f"{args.task}_{clean_model_name}_depth{args.max_depth}_steps{args.max_steps_per_node}_acc{np.mean(test_acc):.4f}.pkl")

        model.llm_engine = None  # Remove engine reference for pickling
        with open(save_model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"[INFO] Model saved to {save_model_path}")

        # Restore llm_engine for further evaluation
        model.llm_engine = llm_api_eval

    if args.do_vis:
        # Set model name
        save_tree_vis_path = os.path.join(args.out_dir, f"{args.task}_{clean_model_name}_depth{args.max_depth}_steps{args.max_steps_per_node}_acc{np.mean(test_acc)}.png")

        # Visualize the trained tree
        print("\n" + "="*50)
        print("GENERATING TREE VISUALIZATION")
        print("="*50)
        
        try:
            # Create visualization with training statistics
            va.visualize_acart_tree(
                root=model.root,
                data=train_data,
                labels=train_labels,
                llm_engine=llm_api_eval,
                max_workers=args.num_threads,
                keyword=keyword,
                data_type=data_type,
                save_path=save_tree_vis_path)
                
        except Exception as e:
            print(f"[ERROR] Failed to generate tree visualization: {e}")
            print("This might be due to missing visualization dependencies (networkx, matplotlib, pygraphviz)")
            print("Try: pip install networkx matplotlib pygraphviz")

    try: 
        ah.cleanup_checkpoints(args.checkpoint_dir, train_run_name)
    except:
        print(f"[WARNING] Did not manage to clean up checkpoints at {args.checkpoint_dir}")

if __name__ == "__main__":
    main()