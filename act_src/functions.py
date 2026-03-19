"""
functions.py
============

Core implementation of the Agentic Classification Tree (ACT). Defines the tree
structure, splitting logic, and recursive training procedure.

This module is imported by act.py and is not meant to be run directly.

This script has 4 Parts:
- Part 1: Helper functions for stability of LLM calls
- Part 2: Functions to evaluate an ACT
- Part 3: Core model classes
- Part 4: Loss and Feedback function
"""

import re
import concurrent.futures
from tqdm import tqdm
import time
from functools import wraps
import textgrad as tg
import numpy as np
import random
import warnings
import function_prompts as fp
import act_helper as ah
warnings.filterwarnings("ignore", category=UserWarning, module="httpx")

try:
    from sklearn.metrics import (
        f1_score,
        precision_recall_fscore_support,
        confusion_matrix,
        matthews_corrcoef,
    )
    SKLEARN_AVAILABLE = True
except Exception:
    print("[WARNING] Unable to import sklearn metrics")
    SKLEARN_AVAILABLE = False

# Part 1 - Helper functions for stability of LLM calls

def retry_with_backoff(max_retries=3, initial_delay=1):
    '''Retry decorator to handel API rate limits'''
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        print(f"[ERROR] Final attempt failed: {e}")
                        raise
                    print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
            raise last_exception  # Should never reach here, but just in case
        return wrapper
    return decorator

class LLMTimeoutError(TimeoutError):
    pass

def call_with_timeout(func, timeout_s, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _executor:
        fut = _executor.submit(func, *args, **kwargs)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            raise LLMTimeoutError(f"LLM call exceeded {timeout_s} seconds")


# Part 2 - Evaluation functions

def eval_sample(item, eval_fn, model, keyword, data_type):
    '''Evaluate one sample using the trained ACART as model. Returns 1 if the correct label was predicted, 0 if the wrong one was predicted.'''
    x, y = item
    is_fallback = False
    is_refused = False
    if not isinstance(x, str) or not isinstance(y, str):
        raise TypeError(f"Expected string for data, got x = {type(x).__name__} and y = {type(y).__name__}")
    
    x_var = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y_var = tg.Variable(str(1 if y == "yes" else 0), requires_grad=False, role_description="correct answer for the query")  # Convert to binary

    try:
        y_int = int(y_var.value) # Get ground truth label for confusion matrix
    except:
        raise ValueError(f"[ERROR] Could not convert label appropriatly")

    try:
        response = model.forward(x_var, keyword, data_type)

        # Extract path length (set by CARTNode.forward); may be None
        path_length = getattr(response, "path_length", None)

        # If the model is currently configured to collect path lengths, log it
        if getattr(model, "_collect_path_lengths", False):
            if not hasattr(model, "_path_lengths") or model._path_lengths is None:
                model._path_lengths = []
            model._path_lengths.append(path_length)

        if response.value.lower() in ['yes', 'no']:
            response.value = str(1 if response.value.lower() == "yes" else 0)  # Convert to binary
        elif response.value.lower() in ['sorry', 'refused']:
            print(f"[WARNING] Ignored an example at evaluation, because model refused to answer, due to security concerns")
            is_refused = True
            return -99, is_fallback, is_refused, y_int
        elif response.value.lower() == 'fallback':
            is_fallback = True
            response.value = str(0) # Return fallback label
        else:
            raise ValueError(f"[WARNING] Response: '{response.value}' is neither yes nor no nor refused nor fallback")
        
        try:
            eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y_var))
            return int(eval_output_variable.value), is_fallback, is_refused, y_int
        except Exception:
            eval_output_variable = eval_fn([x_var, y_var, response])
            eval_output_parsed = eval_fn.parse_output(eval_output_variable)
            print("[WARNING] Eval_sample had to fall back to parse_output and return value eval_sample = ", eval_output_parsed)
            return int(eval_output_parsed), is_fallback, is_refused, y_int
    except Exception as e:
        print(f"[ERROR] Could not evaluate prompt:\n{x[:100]}\n with Label: {y}\n becaue of Error: {e}")
        is_fallback = True
        return 0, is_fallback, is_refused, y_int

def eval_dataset(test_set, eval_fn, model, max_workers, keyword, data_type, max_samples: int=None):
    '''Evaluate ACART model and display accuracy and progress bar'''
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    nbr_fallbacks = 0
    nbr_refused = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            future = executor.submit(eval_sample, sample, eval_fn, model, keyword, data_type)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item, is_fallback, is_refused, _ = future.result()
            nbr_fallbacks += 1 if is_fallback else 0
            nbr_refused += 1 if is_refused else 0
            if acc_item == -99: # Ignore samples, for which the model refused to answer (only issue for Jailbreak dataset)
                continue
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list, nbr_fallbacks, nbr_refused

def eval_dataset_with_labels(test_set, eval_fn, model, num_threads, keyword, data_type, max_samples: int=None):
    """
    Returns:
        correct_list: List[int] with 0/1 correctness per sample
        y_true_list:  List[int or None] ground-truth labels (0/1 if available)
    """
    if max_samples is None:
        max_samples = len(test_set)
    correct_list = []
    y_true_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            future = executor.submit(eval_sample, sample, eval_fn, model, keyword, data_type)
            futures.append(future)
            if len(futures) >= max_samples:
                break

        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            correct, _, _, y_true = future.result()
            correct_list.append(correct)
            y_true_list.append(y_true)
            tqdm_loader.set_description(f"Accuracy: {np.mean(correct_list) if len(correct_list)>0 else 0.0:.4f}")

    return correct_list, y_true_list

def reconstruct_predictions_from_correctness(y_true, correct_list):
    """
    Given ground-truth binary labels y_true (0/1) and a correctness bit per item,
    reconstruct predicted labels:
        if correct -> pred = y_true
        else       -> pred = 1 - y_true
    """
    y_pred = []
    for yt, c in zip(y_true, correct_list):
        if yt is None:
            y_pred.append(None)
        else:
            y_pred.append(yt if c == 1 else 1 - yt)
    return y_pred

def compute_binary_metrics(y_true, y_pred):
    # Filter out any None (in case labels weren’t mapped yet)
    pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt is not None and yp is not None]
    if not pairs:
        return None  # nothing to compute

    y_true_clean, y_pred_clean = zip(*pairs)
    y_true_clean = np.array(y_true_clean, dtype=int)
    y_pred_clean = np.array(y_pred_clean, dtype=int)

    if SKLEARN_AVAILABLE:
        tn, fp, fn, tp = confusion_matrix(y_true_clean, y_pred_clean, labels=[0,1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_clean, y_pred_clean, average="binary", zero_division=0
        )
        mcc = matthews_corrcoef(y_true_clean, y_pred_clean)
    else:
        raise RuntimeError("Sklearn not working!")

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
    }

# Part 3 - Core model classes (CARTNode and CARTAgent)
class CARTNode:
    """A node in a Classification (and potentially Regression) Tree.
    
    This class represents a single decision node or leaf in the CART structure,
    handling both the splitting logic and prediction for classification tasks.
    
    Attributes:
        depth (int): Depth of the node in the tree.
        max_depth (int): Maximum allowed depth for the tree.
        prompt (str): The question/prompt used for splitting at this node.
        is_leaf (bool): Whether this node is a leaf (terminal) node.
        label (str): Prediction label for leaf nodes ('yes' or 'no').
        left (CARTNode): Left child node (for 'yes' answers).
        right (CARTNode): Right child node (for 'no' answers).
    
    Methods:
        forward(x, llm_engine): Process input through the node's decision logic.
    """

    def __init__(self, depth, max_depth, prompt=None, is_leaf=False, label=None):
        self.depth = depth
        self.max_depth = max_depth
        self.prompt = prompt  # The prompt/question at this node
        self.is_leaf = is_leaf
        self.label = label  # Only set for leaf nodes
        self.min_gini = float("inf") # Only set for non leaf nodes
        self.left = None
        self.right = None

    def forward(self, sample, llm_engine, keyword, data_type):
        if self.is_leaf:
            var = tg.Variable(str(self.label), requires_grad=False, role_description="Leaf node label")
            var.path_length = self.depth
            return var
        
        system_prompt = fp.get_node_inference_system_prompt(question=self.prompt, keyword=keyword, data_type=data_type)
        agent_llm = tg.BlackboxLLM(
            engine=llm_engine,
            system_prompt=tg.Variable(
                system_prompt,
                requires_grad=False,
                role_description="System prompt for the model to provide a yes/no answer"
            )
        )
        response = agent_llm(sample).value
        end_of_response = response[-30:].lower()
        think_end = response.lower().find('</think>')
        start_pos = (think_end + 8) if think_end != -1 else 0
        start_of_response = response[start_pos:start_pos + 30].lower()
        
        if "yes" in end_of_response:
            if self.left:
                return self.left.forward(sample, llm_engine, keyword, data_type)
        elif "no" in end_of_response:
            if self.right:
                return self.right.forward(sample, llm_engine, keyword, data_type)
        elif "yes" in start_of_response:
            if self.left:
                return self.left.forward(sample, llm_engine, keyword, data_type)
        elif "no" in start_of_response:
            if self.right:
                return self.right.forward(sample, llm_engine, keyword, data_type)
        elif "sorry" in response.lower() or "refused" in response.lower():
            var = tg.Variable('refused', requires_grad=False, role_description="Security refused label")
            var.path_length = self.depth
            return var
        else:
            reason_for_fallback = 'neither yes or no in response' if not ('yes' in end_of_response or 'no' in end_of_response) else f"Left node exists: '{self.left is not None}' and right node exists: '{self.right is not None}' with end of response: {end_of_response}'"
            n = random.uniform(0, 1)
            if n < 0.5 and self.left:
                print(f"[ERROR] Fallback path left was used in CARTNode.forward() for \n response: {response[:50]} ...  {response[-50:]} \n to input: {sample.value[:50]} \n because of: {reason_for_fallback}")
                return self.left.forward(sample, llm_engine, keyword, data_type)
            elif self.right:
                print(f"[ERROR] Fallback path right was used in CARTNode.forward() for \n response: {response[:50]} ...  {response[-50:]} \n to input: {sample.value[:50]} \n because of: {reason_for_fallback}")
                return self.right.forward(sample, llm_engine, keyword, data_type)
            else:
                print(f"[ERROR] Fallback label was returned in CARTNode.forward() for \n response: {response[:50]} ...  {response[-50:]} \n to input: {sample.value[:50]} \n because of: {reason_for_fallback}")
                var = tg.Variable('fallback', requires_grad=False, role_description="Fallback label")
                var.path_length = self.depth
                return var


class CARTAgent:
    """Main agent for building and using LLM-powered Classification and Regression Trees.
    
    This class manages the entire ACART (Agentic Classification and Regression Tree) model,
    handling tree construction, training via TextGrad optimization, and inference. The tree
    uses language models to make splitting decisions at each node, with prompts optimized
    to minimize Gini impurity.
    
    Attributes:
        llm_engine: The language model engine used for inference and training.
        root (CARTNode): The root node of the decision tree.
    
    Methods:
        build_tree(depth, max_depth): Recursively constructs the tree structure.
        forward(x): Performs inference on input data through the tree.
        fit(...): Trains the tree by optimizing splitting prompts at each node.
    """

    def __init__(self, llm_engine, max_depth):
        self.llm_engine = llm_engine
        self.root = self.build_tree(depth=0, max_depth=max_depth)
        print(f"[INFO] Created CARTAgent model with max_depth {max_depth}.")

    def build_tree(self, depth, max_depth):
        if depth >= max_depth:
            # Create a leaf node (label will be set during training) when depth >= max_depth
            return CARTNode(depth=depth, max_depth=max_depth, is_leaf=True)

        prompt = fp.get_seed_question() # Initialize with unbiased seed question
        node = CARTNode(depth=depth, max_depth=max_depth, prompt=prompt)
        node.left = self.build_tree(depth=depth+1, max_depth=max_depth)
        node.right = self.build_tree(depth=depth+1, max_depth=max_depth)
        return node

    def forward(self, x, keyword, data_type):
        return self.root.forward(sample=x, llm_engine=self.llm_engine, keyword=keyword, data_type=data_type)

    def fit(
        self,
        train_data, train_labels,
        llm_engine, max_steps_per_node,
        max_depth, path, max_workers,
        max_logical_ops, max_examples_per_group,
        keyword, data_type,
        two_stage_training, start_opt_constraints,
        stop_min_samples, min_gini,
        checkpoint_dir, checkpoint_frequency, train_run_name,
        save_checkpoint,
    ):
        fit_cart_llm(
            node=self.root,
            data=train_data,
            labels=train_labels,
            llm_engine=llm_engine,
            max_steps_per_node=max_steps_per_node,
            max_depth=max_depth,
            path=path,
            max_workers=max_workers,
            max_logical_ops=max_logical_ops,
            max_examples_per_group=max_examples_per_group,
            keyword=keyword,
            data_type=data_type,
            two_stage_training=two_stage_training,
            start_opt_constraints=start_opt_constraints,
            stop_min_samples=stop_min_samples,
            min_gini=min_gini,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=checkpoint_frequency,
            train_run_name=train_run_name,
            save_checkpoint=save_checkpoint
        )

# Part 4 - Loss & feedback functions (LLM-driven) -> Purpose: translate split quality into a differentiable (textual) signal and provide actionable feedback to refine the prompt.

@retry_with_backoff(max_retries=3, initial_delay=0.5)
def analyze_group_purity(examples, labels, engine, group_name,
                         current_question, max_examples_per_group,
                         keyword, data_type, path, stop_min_samples):
    '''Give feedback on the quality of the current split'''

    if len(examples) != len(labels):
        raise ValueError(f"[ERROR] examples and labels passed to agp must have the same length, but currently there are {len(examples)} examples and {len(labels)} labels")

    if len(examples) <= stop_min_samples // 2:
        print(f"[INFO] at AGP only {len(examples)} examples, therefor no feedback from {group_name} group")
        return "This group has too few examples. Exclude its feedback, but keep feedback from the other group."
    
    # Build well- and missclassified sample lists with max_n number of samples
    predicted_label = "yes" if "yes" in group_name.lower() else "no"
    mismatched_label = "no" if predicted_label == "yes" else "yes"
    max_n = max_examples_per_group

    well_classified_all = [ex for ex, y in zip(examples, labels) if y == predicted_label]
    misclassified_all = [ex for ex, y in zip(examples, labels) if y == mismatched_label]

    if len(well_classified_all) > max_n:
        well_classified = random.sample(well_classified_all, max_n)
    else:
        well_classified = well_classified_all

    if len(misclassified_all) > max_n:
        misclassified = random.sample(misclassified_all, max_n)
    else:
        misclassified = misclassified_all

    # Get prompt that will be used to get feedback about group purity
    prompt = fp.get_analyze_group_prompt(current_question=current_question,
                                        predicted_label=predicted_label,
                                        well_classified=well_classified,
                                        mismatched_label=mismatched_label,
                                        misclassified=misclassified,
                                        keyword=keyword,
                                        data_type=data_type)
    
    print(f"[INFO] Number of tokens in AGP for path: {path} and {group_name}: {ah.count_tokens(prompt)}")
    
    # Get feedback from LLM and return it, with guardrails for time outs
    def _do_call():
        return engine(prompt)
    try:
        result = call_with_timeout(_do_call, timeout_s=180)
        return result
    except Exception as e:
        print(f"[WARNING] analyze_group_purity timed out or failed in {group_name} at {path}: {e}")
        # Return minimal fallback feedback so training keeps going
        return "No semantic feedback due to model timeout/error. Prefer more precise conditions that separate positives from negatives."

def split_data(max_workers, examples, labels, split_func, prompt_var, step, max_steps_per_node, path, keyword, data_type):
    '''Provided with a split function, this function runs inference on all examples and returns the left and right data groups as well as information about these groups.
    Examples for which no yes or no classification could be extracted are return separatley, but still included in the gini computation.'''
    print(f"[INFO] Splitting {len(examples)} data points")
    yes_examples, no_examples, error_examples = [], [], []
    yes_labels, no_labels, error_labels = [], [], []

    # Inference call -> predict class for each sample and parallelize split_func calls for efficiency
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ex, lab in zip(examples, labels):
            futures.append(executor.submit(split_func, prompt_var, ex, keyword, data_type))
        results = [future.result() for future in futures]

    # Build groups for each class
    for (ex, lab), res in zip(zip(examples, labels), results):
        if res == 'yes':
            yes_examples.append(ex)
            yes_labels.append(lab)
        elif res == 'no':
            no_examples.append(ex)
            no_labels.append(lab)
        elif res == 'error':
            error_examples.append(ex)
            error_labels.append(lab)
        else:
            raise ValueError(f"Split function provided unpossible results {res} for exampls: {ex} and label {lab}")

    # Ensure valid data before calling gini_impurity and format_distribution
    try:
        distribution_yes = ah.format_distribution(yes_labels)
        distribution_no = ah.format_distribution(no_labels)
        gini_yes = ah.gini_impurity(yes_labels)
        gini_no = ah.gini_impurity(no_labels)

        # Compute weighted Gini impurity for the split
        total = len(yes_labels) + len(no_labels) + len(error_labels)
        weighted_gini = ((len(yes_labels) / total) * gini_yes) + ((len(no_labels) / total) * gini_no) + ((len(error_labels) / total) * 0.5)

        if not (0 <= weighted_gini <= 0.5):
            raise ValueError(f"Expected {weighted_gini} to be between 0 and 0.5")
        
        # Add debug print for weighted Gini
        print(f"[INFO] For path: {path} at step {step+1}/{max_steps_per_node} -> Weighted Gini impurity: {weighted_gini}")
        print(f"[INFO] For path: {path} at step {step+1}/{max_steps_per_node} -> {len(error_examples)} / {len(examples)} could not be classified correctly")
    except ValueError as e:
        print(f"[ERROR] Invalid labels encountered: {e} and therefore default labels were returned")
        distribution_yes = "Invalid data"
        distribution_no = "Invalid data"
        gini_yes = 0.5
        gini_no = 0.5
        weighted_gini = 0.5

    return yes_examples, yes_labels, no_examples, no_labels, error_examples, error_labels, distribution_yes, distribution_no, gini_yes, gini_no, weighted_gini


def loss_fn(
    yes_examples, yes_labels,
    no_examples, no_labels,
    distribution_yes, distribution_no,
    gini_yes, gini_no, weighted_gini,
    prompt_var,
    max_logical_ops,
    max_examples_per_group,
    keyword,
    data_type,
    path,
    stop_min_samples,
    engine = None,
    llm_api_eval=None,
):
    '''Based on how the current question split the data this function provides feedback on how to update the current question'''

    # Create FormattedLLMCall object (which when called, sends the filled out prompt format to the llm engine and gives back the answer)
    split_eval = tg.autograd.FormattedLLMCall(
        engine=(engine or llm_api_eval),
        format_string=fp.get_score_format(),
        fields={
            "prompt": None,
            "size_yes": None,
            "size_no": None,
            "distribution_yes": None,
            "distribution_no": None,
            "gini_yes": None,
            "gini_no": None,
            "score": None,
            "feedback_left": None,
            "feedback_right": None,
            "max_logical_ops": None,
            "data_type": None,
            "keyword": None
        },
        system_prompt=tg.Variable(
            "You are a very skilled expert on improving questions used for binary classification. Provide actionable feedback on how to modify the question so that positive class samples are classified by answering 'yes' and negative class samples by answering 'no', while maintaining correct classifications and fixing misclassifications.",
            requires_grad=False,
            role_description="System prompt for the LLM to guide its behavior."
        )
    )

    # Get feedback for left and right leaf to improve prompt in node
    try: 
        sem_feedback_left = analyze_group_purity(
            examples=yes_examples, labels=yes_labels, engine=llm_api_eval,
            group_name="prediction yes", current_question=prompt_var.value,
            max_examples_per_group=max_examples_per_group, keyword=keyword,
            data_type=data_type, path=path, stop_min_samples=stop_min_samples,
        )
        sem_feedback_right = analyze_group_purity(
            examples=no_examples, labels=no_labels, engine=llm_api_eval,
            group_name="prediction no", current_question=prompt_var.value,
            max_examples_per_group=max_examples_per_group, keyword=keyword,
            data_type=data_type, path=path, stop_min_samples=stop_min_samples,
        )

    except ValueError as e:
        print(f"[ERROR] no semantic feedback received because of {e}")
        sem_feedback_left = "No feedback due to error."
        sem_feedback_right = "No feedback due to error."

    # Call FormattedLLMCall object, which returns loss as textual llm output
    feedback_loss = split_eval(inputs={
        "prompt": prompt_var,
        "size_yes": tg.Variable(
            str(len(yes_labels)),
            requires_grad=False,
            role_description="Number of examples in the left group."
        ),
        "size_no": tg.Variable(
            str(len(no_labels)),
            requires_grad=False,
            role_description="Number of examples in the right group."
        ),
        "distribution_yes": tg.Variable(
            distribution_yes,
            requires_grad=False,
            role_description="Class distribution (counts) for the left group."
        ),
        "distribution_no": tg.Variable(
            distribution_no,
            requires_grad=False,
            role_description="Class distribution (counts) for the right group."
        ),
        "gini_yes": tg.Variable(
            f"{gini_yes:.3f}",
            requires_grad=False,
            role_description="Gini impurity for the left group."
        ),
        "gini_no": tg.Variable(
            f"{gini_no:.3f}",
            requires_grad=False,
            role_description="Gini impurity for the right group."
        ),
        "score": tg.Variable(
            f"{weighted_gini:.3f}",
            requires_grad=False,
            role_description="Weighted Gini."
        ),
        "feedback_left": tg.Variable(
            sem_feedback_left,
            requires_grad=False,
            role_description="Semantic feedback for the left group."
        ),
        "feedback_right": tg.Variable(
            sem_feedback_right,
            requires_grad=False,
            role_description="Semantic feedback for the right group."
        ),
        "max_logical_ops": tg.Variable(
            str(max_logical_ops),
            requires_grad=False,
            role_description="Maximum number of logical operators ('and'/'or') allowed in a splitting question."
        ),
        "data_type": tg.Variable(
            ah.remove_article_from_data_type(data_type),
            requires_grad=False,
            role_description="Additional information about the data samples, to guide LLM."
        ),
        "keyword": tg.Variable(
            keyword,
            requires_grad=False,
            role_description="Keyword indicating on what type of information the LLM should focus."
        ),    }, response_role_description="a guide for optimizing the classification question, with the objective of MINIMIZING the weighted Gini impurity of the binary split obtained.")

    # return loss (= feedback on how to improve current question / split) and weighted gini
    return feedback_loss

# Part 5 - Training/optimization loop -> Purpose: learn the splitting question per node so that the dataset becomes purer down the tree.

def fit_cart_llm(
    node, data, labels,
    llm_engine, max_steps_per_node,
    max_depth, path, max_workers,
    max_logical_ops, max_examples_per_group,
    keyword, data_type, two_stage_training,
    start_opt_constraints, stop_min_samples, min_gini,
    checkpoint_dir, checkpoint_frequency, train_run_name,
    save_checkpoint

):
    '''
    Training function for LLM based CART. Starts training at the root node and then trains the children node concurrently and recursively.
    Training of a node is independant of any other node, but the training data a node sees depends on its parent.
    Training a node = optimizing the binary classification prompt at this node. Leaf nodes are not trained, they represent the label given to an input.
    '''

    # Print for debugging and monitoring
    print(f"[INFO] fit_cart_llm: path={path}, node.depth={node.depth}, max_depth={max_depth}, max_n={max_examples_per_group}")

    # Check if node has to be trained or if it is a leaf node
    to_stop, reason_to_stop = ah.stopping_criteria(labels, min_samples=stop_min_samples, max_depth=max_depth, depth=node.depth, min_gini=min_gini)
    if to_stop: # Check if nore qualifies as a leaf node
        node.is_leaf = True
        node.label = ah.majority(labels)  # Assign majority class as the label
        print(f"[INFO] Exiting at leaf. Path: {path}, Node depth: {node.depth}, label: {node.label}, data size: {len(data)} because of {reason_to_stop}")
        return
    
    # Create prompt variable, which is optimized during training (requires_grad = True)
    prompt_var = tg.Variable(
        node.prompt if node.prompt else fp.get_seed_question(),  # Optimizing existing prompt or start from seed question as initial prompt
        requires_grad=True,
        role_description=fp.get_role_of_var_to_optimize(),
    )

    # Initialize TDG optimizer for the prompt variable and define constraints for constrained optimization of the node prompt
    training_stage=start_opt_constraints
    optimizer = tg.TGD(
        [prompt_var],
        constraints=fp.get_prompt_constraints(keyword=keyword, max_logical_ops=max_logical_ops, training_stage=training_stage),
        verbose=0
    )
    print(f"[INFO] Created optimizer for {path} with initial prompt: {prompt_var.value} and training stage: {training_stage}")

    # Set if training is done in two stages
    switch_step = max_steps_per_node // 2
    print(f"[INFO] Two stage training is set to {two_stage_training}" + (f" and will switch after step {switch_step} for path: {path}" if two_stage_training else ""))

    # Initialize variables used during training
    best_prompt = prompt_var.value # Tracks prompt with the lowest gini
    lowest_gini = float("inf")
    weighted_gini = float("inf")

    # Helper function for infering over the data during training
    @retry_with_backoff(max_retries=3, initial_delay=0.5)
    def evaluate_with_agent(prompt, context, keyword, data_type):

        system_prompt = fp.get_node_inference_system_prompt(question=prompt.value, keyword=keyword, data_type=data_type)

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
            result = call_with_timeout(_do_call, timeout_s=180)
        except Exception as e:
            print(f"[WARNING] evaluate_with_agent timeout/error at {path}: {e}")
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

    # Check for existing checkpoint
    start_step = 0  # Start from 0 if no intermediate checkpoint exists
    resume_children_only = False
    var_wont_be_def_init = False
    training_stopped_early = False
    if checkpoint_dir:
        checkpoint_path = ah.get_checkpoint_path(checkpoint_dir, train_run_name, path, 'latest')
        checkpoint = ah.load_checkpoint(checkpoint_path)

        if checkpoint:
            start_step = checkpoint['step_info'].get('current_step', 0) + 1
            lowest_gini = checkpoint['step_info'].get('lowest_gini', float("inf"))
            best_prompt = checkpoint['step_info'].get('best_prompt', node.prompt or fp.get_seed_question())
            prompt_var.set_value(checkpoint['step_info'].get('best_prompt', fp.get_seed_question()))
            var_wont_be_def_init = True
            if prompt_var.value == fp.get_seed_question():
                print(f"[WARNING] Checkpoint set prompt to seed question")
            # If node finished its optimization previously, we still need to split & recurse
            if start_step >= max_steps_per_node:
                print(f"[CHECKPOINT] Node optimization already completed at {path}; resuming children.")
                resume_children_only = True
            print(f"[CHECKPOINT] Prompt for path {path} is now {prompt_var.value}")
        if checkpoint and start_step >=max_steps_per_node and not resume_children_only:
            raise RuntimeError(f"[ERROR] During the loading of a checkpoint in training an unpossible combionation occured for checkpoint: {checkpoint_path}.")

    # Training loop for one node
    for step in range(start_step, max_steps_per_node):

        if resume_children_only: # skip training this node if it was already fully trained in checkpoint
            break

        if two_stage_training and step == switch_step and 'exploring' in training_stage:
            # Create new optimizer with exploiting constraints
            training_stage = "exploiting"
            optimizer = tg.TGD(
                [prompt_var],
                constraints=fp.get_prompt_constraints(keyword=keyword, max_logical_ops=max_logical_ops, training_stage=training_stage),
                verbose=1
            )
            print(f"[INFO] Switching from exploring to exploiting at step {step+1}")

        optimizer.zero_grad()
        print("="*40)
        print(f"[STEP {step+1}/{max_steps_per_node}] Path: {path}")
        print(f"[INFO] Prompt: {prompt_var.value}")
        
        # Split data using current question and calculate gini
        yes_examples, yes_labels, no_examples, no_labels, error_examples, error_labels, distribution_yes, distribution_no, gini_yes, gini_no, weighted_gini = split_data(max_workers=max_workers,
                                                                                                                                    examples=data,
                                                                                                                                    labels=labels,
                                                                                                                                    split_func=evaluate_with_agent,
                                                                                                                                    prompt_var=prompt_var,
                                                                                                                                    step=step,
                                                                                                                                    max_steps_per_node=max_steps_per_node,
                                                                                                                                    path=path,
                                                                                                                                    keyword=keyword,
                                                                                                                                    data_type=data_type)

        if weighted_gini <= lowest_gini or best_prompt == fp.get_seed_question(): # Get rid of the seed question anyways
            lowest_gini = weighted_gini
            best_prompt = prompt_var.value
            best_yes_examples, best_yes_labels, best_no_examples, best_no_labels, best_error_examples, best_error_labels = yes_examples, yes_labels, no_examples, no_labels, error_examples, error_labels
            best_distribution_yes, best_distribution_no, best_gini_yes, best_gini_no, best_weighted_gini = distribution_yes, distribution_no, gini_yes, gini_no, weighted_gini
        
        if best_prompt != fp.get_seed_question() and weighted_gini == lowest_gini and weighted_gini < min_gini:
            training_stopped_early = True
            print(f"[INFO] Prompt Optimization for path: {path} stopped at step {step+1} / {max_steps_per_node}, because the gini of the best question is {weighted_gini} and therefor below the min gini of {min_gini}")
            break

        # During exploiting stage, always optimize the best current prompt
        if "exploiting" in training_stage:
            prompt_var.set_value(best_prompt)
            yes_examples, yes_labels, no_examples, no_labels, error_examples, error_labels = best_yes_examples, best_yes_labels, best_no_examples, best_no_labels, best_error_examples, best_error_labels
            distribution_yes, distribution_no, gini_yes, gini_no, weighted_gini = best_distribution_yes, best_distribution_no, best_gini_yes, best_gini_no, best_weighted_gini
            print(f"[INFO] Prompt var was set to: {prompt_var.value}")

        # Calculate loss (loss = feedback on the performance of the current splitting question)
        loss = loss_fn(
            yes_examples=yes_examples, yes_labels=yes_labels,
            no_examples=no_examples, no_labels=no_labels,
            distribution_yes=distribution_yes, distribution_no=distribution_no,
            gini_yes=gini_yes, gini_no=gini_no, weighted_gini=weighted_gini,
            prompt_var=prompt_var,
            max_logical_ops=max_logical_ops,
            max_examples_per_group=max_examples_per_group,
            keyword=keyword,
            data_type=data_type,
            path=path,
            stop_min_samples=stop_min_samples,
            engine = llm_engine,
            llm_api_eval=llm_engine,
        )

        try:
            loss.backward()
            optimizer.step() # This step changes the prompt_var.value (analogous to updating the weights)
        except (ValueError, IndexError) as e:
            print(f"[WARNING] Optimization failed at step {step+1}/{max_steps_per_node} for path {path}")
            print(f"[WARNING] Error type: {type(e).__name__}: {str(e)[:200]}...")
            print(f"[WARNING] Keeping current prompt and continuing to next step...")
        except Exception as e:
            print(f"[ERROR] Unexpected error during optimization at step {step+1}/{max_steps_per_node} for path {path}")
            print(f"[ERROR] {type(e).__name__}: {str(e)[:200]}...")
            print(f"[WARNING] Keeping current prompt and continuing to next step...")
        
        # CHECKPOINTING
        if checkpoint_dir and save_checkpoint and (step + 1) % checkpoint_frequency == 0:
            checkpoint_info = {
                'current_step': step,
                'lowest_gini': lowest_gini,
                'best_prompt': best_prompt,
            }
            checkpoint_path = ah.get_checkpoint_path(checkpoint_dir, train_run_name, path, 'latest')
            ah.save_checkpoint(node, checkpoint_path, checkpoint_info)

    # Calculate gini for the last update of the question
    if not training_stopped_early:
        yes_examples, yes_labels, no_examples, no_labels, error_examples, error_labels, distribution_yes, distribution_no, gini_yes, gini_no, final_gini = split_data(max_workers=max_workers,
                                                                                                                                        examples=data,labels=labels,
                                                                                                                                        split_func=evaluate_with_agent,
                                                                                                                                        prompt_var=prompt_var,
                                                                                                                                        step=max_steps_per_node-1,
                                                                                                                                        max_steps_per_node=max_steps_per_node,
                                                                                                                                        path=path, keyword=keyword,
                                                                                                                                        data_type=data_type)
    
    if resume_children_only or ('best_yes_examples' not in locals() and var_wont_be_def_init): # If the prompt stored in the node was laoded from a checkpoint, these variables do not necesseraly exist
        best_yes_examples, best_yes_labels, best_no_examples, best_no_labels, best_error_examples, best_error_labels = yes_examples, yes_labels, no_examples, no_labels, error_examples, error_labels
        best_distribution_yes, best_distribution_no, best_gini_yes, best_gini_no, best_weighted_gini = distribution_yes, distribution_no, gini_yes, gini_no, weighted_gini

    if not training_stopped_early and final_gini <= lowest_gini: # Update split variables if the last split was the best
        lowest_gini = final_gini
        best_prompt = prompt_var.value
        best_yes_examples, best_yes_labels, best_no_examples, best_no_labels, best_error_examples, best_error_labels = yes_examples, yes_labels, no_examples, no_labels, error_examples, error_labels
        best_distribution_yes, best_distribution_no, best_gini_yes, best_gini_no, best_weighted_gini = distribution_yes, distribution_no, gini_yes, gini_no, weighted_gini

    node.prompt = best_prompt # Store best prompt in node
    node.min_gini = lowest_gini
    print(f"[INFO] Best prompt selected for path {path}: {best_prompt} with Gini {lowest_gini}")

    if checkpoint_dir and save_checkpoint:
        checkpoint_info = {
            'current_step': max_steps_per_node - 1,
            'lowest_gini': lowest_gini,
            'best_prompt': best_prompt,
        }
        checkpoint_path = ah.get_checkpoint_path(checkpoint_dir, train_run_name, path, 'latest')
        ah.save_checkpoint(node, checkpoint_path, checkpoint_info)


    # Use training data split from best prompt for left and right node
    left_data, right_data = best_yes_examples, best_no_examples
    left_labels, right_labels = best_yes_labels, best_no_labels

    if isinstance(best_error_examples, list) and isinstance(best_error_labels, list) and all(item in ['yes', 'no'] for item in best_error_labels):
        random.shuffle(best_error_examples)
        random.shuffle(best_error_labels)
    else:
        raise ValueError(f"[ERROR] Error examples and error label list have an unexpected behaviour. They are {type(best_error_examples)} and {type(best_error_labels)}.")
    
    if len(best_error_examples) == len(best_error_labels):
        mid_point = len(best_error_examples) // 2
    else:
        raise ValueError(f"[ERROR] In fit_car_llm for path {path} the length of the error lists are not the same")
    
    left_data.extend(best_error_examples[:mid_point])
    right_data.extend(best_error_examples[mid_point:])
    left_labels.extend(best_error_labels[:mid_point])
    right_labels.extend(best_error_labels[mid_point:])

    if len(left_data) != len(left_labels) or len(right_data) != len(right_labels) or not all(item in ['yes', 'no'] for item in left_labels) or not all(item in ['yes', 'no'] for item in right_labels):
        raise ValueError(f"[ERROR] For path {path} the final constructed data subsets have an issue!")

    print(f"[INFO] left data size: {len(left_data)} and right: {len(right_data)}")

    node.left = CARTNode(depth=node.depth + 1, max_depth=max_depth)
    node.right = CARTNode(depth=node.depth + 1, max_depth=max_depth)

    # Train left and right child nodes in parallel
    child_workers = max(1, max_workers // 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(fit_cart_llm, node.left, left_data, left_labels, llm_engine,
                            max_steps_per_node, max_depth, f"{path}/left", child_workers,
                            max_logical_ops, max_examples_per_group, keyword, data_type,
                            two_stage_training, start_opt_constraints, stop_min_samples, min_gini,
                            checkpoint_dir, checkpoint_frequency, train_run_name, save_checkpoint),
            executor.submit(fit_cart_llm, node.right, right_data, right_labels, llm_engine,
                            max_steps_per_node, max_depth, f"{path}/right", child_workers,
                            max_logical_ops, max_examples_per_group, keyword, data_type,
                            two_stage_training, start_opt_constraints, stop_min_samples, min_gini,
                            checkpoint_dir, checkpoint_frequency, train_run_name, save_checkpoint)
        ]
        # Check for exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] Failed to train child node at path {path}: {e}")