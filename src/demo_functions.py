"""
demo_functions.py
============

Core implementation of the Agentic Classification Tree (ACT). Defines the tree
structure, splitting logic, and recursive training procedure.

This module is imported by demo_helper.py and is not meant to be run directly.

Code has 6 Parts, Ctrl + F and Part 3 will lead you to the start of Part 3
"""

import re
import concurrent.futures
from tqdm import tqdm
import time
from functools import wraps
import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import textwrap
import tiktoken
import warnings
import demo_prompts as fp
warnings.filterwarnings("ignore", category=UserWarning, module="httpx")
from collections import Counter
import matplotlib.pyplot as plt


# Part 1 - General helper functions

def set_seed(seed):
    np.random.seed(seed)
    import random
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
    '''Returns majority class of a list of yes/no labels'''
    print("[INFO] label yes count: ", labels.count("yes"), " and label no count: ", labels.count("no"))
    return "yes" if labels.count("yes") >= labels.count("no") else "no"

def wrap_label(text, width=40):
    return "\n".join(textwrap.wrap(text, width)) if text else ""

def stopping_criteria(labels, min_gini, min_samples=10, max_depth=None, depth=0):
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

# Part 2 - Evaluation helpers (for accuracy) -> Purpose: compute accuracy on train/val/test.

def eval_sample(item, eval_fn, model, keyword, data_type):
    '''Evaluate one sample using the trained ACT model. Returns 1 if the correct label was predicted, 0 if the wrong one was predicted.'''
    x, y = item
    if not isinstance(x, str) or not isinstance(y, str):
        raise TypeError(f"Expected string for data, got x = {type(x).__name__} and y = {type(y).__name__}")
    
    x_var = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y_var = tg.Variable(str(1 if y == "yes" else 0), requires_grad=False, role_description="correct answer for the query")  # Convert to binary

    try:
        response = model.forward(x_var, keyword, data_type)
        response.value = str(1 if response.value.lower() == "yes" else 0)  # Convert to binary
        
        try:
            eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y_var))
            return int(eval_output_variable.value)
        except Exception:
            eval_output_variable = eval_fn([x_var, y_var, response])
            eval_output_parsed = eval_fn.parse_output(eval_output_variable)
            print("[WARNING] return value in eval_sample function = ", eval_output_parsed, " had to be parsed differently, but code execution could still continue.")
            return int(eval_output_parsed)
    except Exception as e:
        print(f"[ERROR] Could not evaluate prompt:\n{x}\n with Label: {y}\n becaue of Error: {e} \n Evaluating continued despite error")
        return 0

def eval_dataset(test_set, eval_fn, model, keyword, data_type, max_samples: int=None, max_workers=2):
    '''Evaluate ACT model and display accuracy and progress bar'''
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            future = executor.submit(eval_sample, sample, eval_fn, model, keyword, data_type)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list

# Part 3 - Core model classes (CARTNode and CARTAgent)
class CARTNode:
    """A node in a Classification and Regression Tree.
    
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

    def __init__(self, depth=0, max_depth=3, prompt=None, is_leaf=False, label=None):
        self.depth = depth
        self.max_depth = max_depth
        self.prompt = prompt  # The prompt/question at this node
        self.is_leaf = is_leaf
        self.label = label  # Only set for leaf nodes
        self.left = None
        self.right = None

    def forward(self, sample, llm_engine, keyword, data_type):
        if self.is_leaf:
            return tg.Variable(str(self.label), requires_grad=False, role_description="Leaf node label")
        
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

        # Remove <R1> and </R1> tags, keep content inside
        response = re.sub(r"</?R1>", "", response).strip()
        
        if "yes" in response[-30:].lower():
            if self.left:
                return self.left.forward(sample, llm_engine, keyword, data_type)
        else:
            if self.right:
                return self.right.forward(sample, llm_engine, keyword, data_type)
            
        print("[ERROR] Fallback label was returned in CARTNode.forward() for response: ", response)
        return tg.Variable("no", requires_grad=False, role_description="Fallback label")


class CARTAgent:
    """Main agent for building and using LLM-powered Classification and Regression Trees.
    
    This class manages the entire ACT (Agentic Classification Tree) model,
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

    def __init__(self, llm_engine, max_depth=3):  # Use a default value
        self.llm_engine = llm_engine
        self.root = self.build_tree(depth=0, max_depth=max_depth)

    def build_tree(self, depth, max_depth):
        if depth >= max_depth:
            # Create a leaf node (label will be set during training) when depth >= max_depth
            return CARTNode(depth=depth, max_depth=max_depth, is_leaf=True)

        prompt = fp.get_seed_question() # Initialize with unbiased seed question
        node = CARTNode(depth=depth, max_depth=max_depth, prompt=prompt)
        node.left = self.build_tree(depth+1, max_depth)
        node.right = self.build_tree(depth+1, max_depth)
        return node

    def forward(self, x, keyword, data_type):
        return self.root.forward(sample=x, llm_engine=self.llm_engine, keyword=keyword, data_type=data_type)

    def fit(
        self,
        train_data,
        train_labels,
        llm_engine,
        max_steps_per_node,
        max_depth,
        path,
        max_workers,
        max_logical_ops,
        max_examples_per_group,
        keyword,
        data_type,
        min_gini,
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
            min_gini=min_gini,
        )

# Part 4 - Loss & feedback functions (LLM-driven) -> Purpose: translate split quality into a differentiable (textual) signal and provide actionable feedback to refine the prompt.

@retry_with_backoff(max_retries=3, initial_delay=0.5)
def analyze_group_purity(examples, labels, engine, group_name, current_question, max_examples_per_group, keyword):
    '''Give feedback on the quality of the current split'''

    if len(examples) != len(labels):
        raise ValueError(f"[ERROR] examples and labels passed to agp must have the same length, but currently there are {len(examples)} examples and {len(labels)} labels")

    if len(examples) <= 5:
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
                                        keyword=keyword)

    # Get feedback from LLM and return it
    result = engine(prompt)
    return result

def split_data(max_workers, examples, labels, split_func, prompt_var, step, max_steps_per_node, path, keyword, data_type):
    '''Provided with a split function, this function runs inference on all examples and returns the left and right data groups as well as information about these groups'''

    yes_examples, no_examples = [], []
    yes_labels, no_labels = [], []

    # Inference call -> predict class for each sample and parallelize split_func calls for efficiency
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ex, lab in zip(examples, labels):
            futures.append(executor.submit(split_func, prompt_var, ex, keyword, data_type))
        results = [future.result() for future in futures]

    # Build groups for each class
    for (ex, lab), is_yes in zip(zip(examples, labels), results):
        if is_yes:
            yes_examples.append(ex)
            yes_labels.append(lab)
        else:
            no_examples.append(ex)
            no_labels.append(lab)

    # Ensure valid data before calling gini_impurity and format_distribution
    try:
        distribution_yes = format_distribution(yes_labels)
        distribution_no = format_distribution(no_labels)
        gini_yes = gini_impurity(yes_labels)
        gini_no = gini_impurity(no_labels)

        # Compute weighted Gini impurity for the split
        total = len(yes_labels) + len(no_labels)
        weighted_gini = ((len(yes_labels) / total) * gini_yes) + ((len(no_labels) / total) * gini_no)

        if not (0 <= weighted_gini <= 0.5):
            raise ValueError(f"Expected {weighted_gini} to be between 0 and 0.5")
        
        # Print weighted Gini for current optimization step (include paths information because of concurrency)
        print(f"[INFO] For path: {path} at step {step+1}/{max_steps_per_node} -> Weighted Gini impurity: {weighted_gini}")

    except ValueError as e:
        print(f"[ERROR] Invalid labels encountered: {e} and therefore default labels were returned")
        distribution_yes = "Invalid data"
        distribution_no = "Invalid data"
        gini_yes = 0.5
        gini_no = 0.5
        weighted_gini = 0.5

    return yes_examples, yes_labels, no_examples, no_labels, distribution_yes, distribution_no, gini_yes, gini_no, weighted_gini


def loss_fn(
    yes_examples, yes_labels,
    no_examples, no_labels,
    distribution_yes, distribution_no,
    gini_yes, gini_no, weighted_gini,
    prompt_var,
    max_logical_ops,
    max_examples_per_group,
    keyword,
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
            examples=yes_examples, labels=yes_labels, engine=llm_api_eval, group_name="prediction yes", current_question=prompt_var.value, max_examples_per_group=max_examples_per_group, keyword=keyword
        )
        sem_feedback_right = analyze_group_purity(
            examples=no_examples, labels=no_labels, engine=llm_api_eval, group_name="prediction no", current_question=prompt_var.value, max_examples_per_group=max_examples_per_group, keyword=keyword
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
        "keyword": tg.Variable(
            keyword,
            requires_grad=False,
            role_description="Keyword indicating on what type of information the LLM should focus."
        ),    }, response_role_description="a guide for optimizing the classification question, with the objective of MINIMIZING the weighted Gini impurity of the binary split obtained.")

    # return loss (= feedback on how to improve current question / split) and weighted gini
    return feedback_loss

# Part 5 - Training/optimization loop -> Purpose: learn the splitting question per node so that the dataset becomes purer down the tree.

def fit_cart_llm(
    node,
    data,
    labels,
    llm_engine,
    max_steps_per_node,
    max_depth,
    path,
    max_workers,
    max_logical_ops,
    max_examples_per_group,
    keyword,
    data_type,
    min_gini
):
    '''
    Training function for LLM based CART. Starts training at the root node and then trains the children node concurrently and recursively.
    Training of a node is independant of any other node, but the training data a node sees depends on its parent.
    Training a node = optimizing the binary classification prompt at this node. Leaf nodes are not trained, they represent the label given to an input.
    '''

    # Print for debugging and monitoring
    print(f"[INFO] fit_cart_llm: path={path}, node.depth={node.depth}, max_depth={max_depth}, max_n={max_examples_per_group}")

    # Check if node has to be trained or if it is a leaf node
    to_stop, reason_to_stop = stopping_criteria(labels, min_samples=5, max_depth=max_depth, depth=node.depth, min_gini=min_gini)
    if to_stop: # Check if nore qualifies as a leaf node
        node.is_leaf = True
        node.label = majority(labels)  # Assign majority class as the label
        print(f"[INFO] Exiting at leaf. Path: {path}, Node depth: {node.depth}, label: {node.label}, data size: {len(data)} because of {reason_to_stop}")
        return
    
    # Create prompt variable, which is optimized during training (requires_grad = True)
    prompt_var = tg.Variable(
        node.prompt if node.prompt else fp.get_seed_question(),  # Optimizing existing prompt or start from seed question as initial prompt
        requires_grad=True,
        role_description=fp.get_role_of_var_to_optimize(),
    )

    # Initialize TDG optimizer for the prompt variable and define constraints for constrained optimization of the node prompt
    optimizer = tg.TGD(
        [prompt_var],
        constraints=fp.get_prompt_constraints(keyword=keyword, max_logical_ops=max_logical_ops),
        verbose=0
    )

    # Initialize variables used during training
    best_prompt = prompt_var.value # Tracks prompt with the lowest gini
    lowest_gini = float("inf")
    weighted_gini = float("inf")

    # Helper function for infering over the data during training
    @retry_with_backoff(max_retries=3, initial_delay=0.5)
    def evaluate_with_agent(prompt, context, keyword, data_type):

        system_prompt = fp.get_node_inference_system_prompt(question=prompt.value, keyword=keyword, data_type=data_type)

        # Initiallize LLM Agent used that classifies the samples using the prompt in the node
        agent_llm = tg.BlackboxLLM( # TODO Used to be deepseek
            engine=llm_engine,
            system_prompt=tg.Variable(
                system_prompt,
                requires_grad=False,
                role_description="System prompt for the model to provide a yes/no answer"
            )
        )

        # Use agent_llm passed as argument, not global
        result = agent_llm(
            tg.Variable(
                context,
                requires_grad=False,
                role_description="prompt that instructs the LLM to answer the question with yes/no based on the context provided."
            )
        ).value.lower()
        return "yes" in result[-30:].lower() # Returns True if "yes" in result, False otherwise TODO is this robust enough?



    # Training loop for one node
    for step in range(max_steps_per_node):

        optimizer.zero_grad()
        print("="*40)
        print(f"[STEP {step+1}/{max_steps_per_node}] Path: {path}")
        print(f"[INFO] Prompt: {prompt_var.value}")
        
        # Split data using current question and calculate gini
        yes_examples, yes_labels, no_examples, no_labels, distribution_yes, distribution_no, gini_yes, gini_no, weighted_gini = split_data(max_workers=max_workers,
                                                                                                                                    examples=data,
                                                                                                                                    labels=labels,
                                                                                                                                    split_func=evaluate_with_agent,
                                                                                                                                    prompt_var=prompt_var,
                                                                                                                                    step=step,
                                                                                                                                    max_steps_per_node=max_steps_per_node,
                                                                                                                                    path=path,
                                                                                                                                    keyword=keyword,
                                                                                                                                    data_type=data_type)
        
        if weighted_gini <= lowest_gini or best_prompt == fp.get_seed_question(): # TODO Check if it makes sense to get rid of the seed question anyways
            lowest_gini = weighted_gini
            best_prompt = prompt_var.value
            best_yes_examples, best_yes_labels, best_no_examples, best_no_labels = yes_examples, yes_labels, no_examples, no_labels
            best_distribution_yes, best_distribution_no, best_gini_yes, best_gini_no, best_weighted_gini = distribution_yes, distribution_no, gini_yes, gini_no, weighted_gini
        
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
            engine = llm_engine,
            llm_api_eval=llm_engine,
        )

        loss.backward()
        optimizer.step() # This step changes the prompt_var.value (analogous to updating the weights)

    # Calculate gini for the last update of the question
    yes_examples, yes_labels, no_examples, no_labels, distribution_yes, distribution_no, gini_yes, gini_no, final_gini = split_data(max_workers=max_workers,
                                                                                                                                    examples=data,labels=labels,
                                                                                                                                    split_func=evaluate_with_agent,
                                                                                                                                    prompt_var=prompt_var,
                                                                                                                                    step=max_steps_per_node-1,
                                                                                                                                    max_steps_per_node=max_steps_per_node,
                                                                                                                                    path=path, keyword=keyword,
                                                                                                                                    data_type=data_type)
    if final_gini <= lowest_gini:
        lowest_gini = final_gini
        best_prompt = prompt_var.value
        best_yes_examples, best_yes_labels, best_no_examples, best_no_labels = yes_examples, yes_labels, no_examples, no_labels
        best_distribution_yes, best_distribution_no, best_gini_yes, best_gini_no, best_weighted_gini = distribution_yes, distribution_no, gini_yes, gini_no, weighted_gini


    node.prompt = best_prompt # Store best prompt in node
    print(f"[INFO] Best prompt selected for path {path}: {best_prompt} with Gini {lowest_gini}")

    # Split training data for left and right node -> TODO should we just re-use the data split from before?
    left_data, right_data = best_yes_examples, best_no_examples
    left_labels, right_labels = best_yes_labels, best_no_labels

    print(f"[INFO] left data size: {len(left_data)} and right: {len(right_data)}")

    node.left = CARTNode(depth=node.depth + 1, max_depth=max_depth)
    node.right = CARTNode(depth=node.depth + 1, max_depth=max_depth)

    # Train left and right child nodes in parallel
    child_workers = max(1, max_workers // 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(fit_cart_llm, node.left, left_data, left_labels, llm_engine,
                            max_steps_per_node, max_depth, f"{path}/left", child_workers,
                            max_logical_ops, max_examples_per_group, keyword, data_type, min_gini),
            executor.submit(fit_cart_llm, node.right, right_data, right_labels, llm_engine,
                            max_steps_per_node, max_depth, f"{path}/right", child_workers,
                            max_logical_ops, max_examples_per_group, keyword, data_type, min_gini)
        ]
        # Check for exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise any exception that occurred
            except Exception as e:
                print(f"[ERROR] Failed to train child node at path {path}: {e}")

# Part 6 - Visualize ACT tree

def plot_act_tree(root, data, labels, llm_engine, max_workers, keyword, data_type):
    """
    Create a NetworkX graph representation of the ACT tree with training statistics.
    
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
        gini = gini_impurity(labels_subset) if labels_subset else 0.0
        
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
            wrapped_prompt = wrap_label(node.prompt, width=30)
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
            @retry_with_backoff(max_retries=3, initial_delay=0.5)
            def evaluate_with_agent(prompt, context, keyword, data_type):

                system_prompt = fp.get_node_inference_system_prompt(question=prompt, keyword=keyword, data_type=data_type)

                # Initiallize LLM Agent used that classifies the samples using the prompt in the node
                agent_llm = tg.BlackboxLLM( # TODO Used to be deepseek
                    engine=llm_engine,
                    system_prompt=tg.Variable(
                        system_prompt,
                        requires_grad=False,
                        role_description="System prompt for the model to provide a yes/no answer"
                    )
                )

                # Use agent_llm passed as argument, not global
                result = agent_llm(
                    tg.Variable(
                        context,
                        requires_grad=False,
                        role_description="prompt that instructs the LLM to answer the question with yes/no based on the context provided."
                    )
                ).value.lower()
                return "yes" in result[-30:].lower() # Returns True if "yes" in result, False otherwise TODO is this robust enough?


            left_data, left_labels, right_data, right_labels, _, _, _, _, _ = split_data(max_workers=max_workers,
                                                                                        examples=data,
                                                                                        labels=labels,
                                                                                        split_func=evaluate_with_agent,
                                                                                        prompt_var=node.prompt,
                                                                                        step=current_id,
                                                                                        max_steps_per_node="Plotting",
                                                                                        path="at plotting time",
                                                                                        keyword=keyword,
                                                                                        data_type=data_type)

            
            # Recursively add child nodes
            add_node_to_graph(node.left, left_data, left_labels, current_id, "Yes")
            add_node_to_graph(node.right, right_data, right_labels, current_id, "No")
        
        return current_id
    
    # Start building the graph from root
    add_node_to_graph(root, data, labels)
    return graph


def visualize_act_tree(root, data, labels, llm_engine, max_workers, keyword, data_type, save_path=None):
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
    graph = plot_act_tree(root, data=data, labels=labels, llm_engine=llm_engine, max_workers=max_workers, keyword=keyword, data_type=data_type)
    node_labels = nx.get_node_attributes(graph, 'label')
    
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
    plt.title("ACT Decision Tree\n(Green=Yes prediction, Red=No prediction, Blue/Yellow/Pink=Internal nodes by Gini)", 
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