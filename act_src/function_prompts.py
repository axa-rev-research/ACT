"""
function_prompts.py
==========

Contains the prompts used during training and evaluation for the Agentic Classification Tree (ACT) scripts.

Usage:
    - Is imported by ACART_1.py and functions.py and used there
"""

def remove_article_from_data_type(data_type):
    """Remove leading article (a, an, the) from a string."""
    words = data_type.split(maxsplit=1)
    if len(words) > 1 and words[0].lower() in ['a', 'an', 'the']:
        return words[1]
    else:
        raise ValueError(f"Data type does not start with an article (a, an, the) as expected: '{data_type}'")

def get_seed_question():
    '''Default prompt to initalize a node'''
    return "Based on the provided context, does this example belong to the positive class? (yes/no)"

def get_score_format():
    '''format of prompt used in loss_fn to get the loss'''
    
    return """
Your task is to give feedback about a binary classification question, such that the question can be improved.

The samples being classified using the yes/no question are {data_type}.

This is the current yes/no question used for binary classification that needs to be optimized: <VARIABLE>{prompt}</VARIABLE>

Based on the information provided below, suggest how to modify the question so that:
- All positive class samples are classified by answering 'yes' to the question
- All negative class samples are classified by answering 'no' to the question
- Correctly classified samples remain correctly classified
- Misclassified samples change their answers to become correctly classified

Information about the group for which the model predicted "YES":
- Number of samples: {size_yes}
- True label distribution: {distribution_yes}
- Purity (Gini): {gini_yes}
- Feedback on classification errors and successes: {feedback_left}

Information about the group for which the model predicted "NO":
- Number of samples: {size_no}
- True label distribution: {distribution_no}
- Purity (Gini): {gini_no}
- Feedback on classification errors and successes: {feedback_right}

WEIGHTED_GINI Score: <WEIGHTED_GINI>{score}</WEIGHTED_GINI>

If the weighted Gini score is poor (> 0.45), your feedback must suggest a completely different question that focuses on different {keyword}s of the examples than the current question.

Note: A Gini score of 0 indicates perfectly homogeneous groups (ideal). A score of 0.5 means groups contain equal mix of yes/no answers (worst case). The goal is to minimize the Gini score.

If one group has significantly more samples than the other, focus more on the larger group's feedback. Focus your feedback on at most 2 {keyword}s.

Give clear, concise, and actionable feedback on how to improve the question. Keep your answer short.
"""

def get_prompt_constraints(keyword, max_logical_ops, training_stage):
    '''Constraints used during prompt optimization'''
    
    if max_logical_ops == None:
        logic_rule = ""
    elif max_logical_ops == 0:
        logic_rule = "Do not use 'and' or 'or' in the question."
    else:
        logic_rule = f"Use at most {max_logical_ops} 'and' or 'or' in the question."

    if training_stage == 'basic':
        return [
            "Write a question that is as simple as possible.  Use only one idea.",
            "It should be answerable with 'yes' or 'no' only.",
            "The question must finish with (yes/no)?",
            logic_rule,
            "Do not use vague words like 'could', 'might', or 'possibly'.",
            "Do not use blanks or tags like '___' or <...>.",
            ]
    
    elif training_stage == "exploring":
        return [
            "The question has to be clear and easy.",
            f"The question must focus on at most 2 {keyword}s.",
            f"The question has to focus on at least one {keyword} different from the previous one.",
            "The content of the new question has to be significantly different from the current one.",
            logic_rule,
            "The question has to be answerable with 'yes' or 'no' only.",
            "The question must finish with (yes/no)?",
            "Do not use vague words like 'could', 'might', or 'possibly'.",
            "Do not use blanks or tags like '___' or <...>."
        ]
    
    elif training_stage == "exploiting":
        return [
            "The question has to be clear and easy.",
            "The new question should not just be a rewritten version of the current one.",
            logic_rule,
            "The question has to be answerable with 'yes' or 'no' only.",
            "The question must finish with (yes/no)?",
            "Do not use vague words like 'could', 'might', or 'possibly'.",
            "Do not use blanks or tags like '___' or <...>."
        ]
    
    elif training_stage == "demo_notebook":
        return [
            "The question has to be clear and easy.",
            f"The question must focus on at most 2 {keyword}s.",
            logic_rule,
            f"The question has to focus on at least one {keyword} different from the previous one.",
            "The content of the new question has to be significantly different from the current one.",
            "The question has to be answerable with 'yes' or 'no' only.",
            "The question must finish with (yes/no)?",
            "Do not use vague words like 'could', 'might', or 'possibly'.",
            "Do not use blanks or tags like '___' or <...>."
        ]
        
    else:
        raise ValueError(f"[ERROR] training stage {training_stage} is not a valid option")

def get_node_inference_system_prompt(question, keyword = "characteristic", data_type = "a text"):
    """System prompt used for inference on one node"""
    sys_prompt = f"""Given {data_type}, your job is to answer the following question:
    
    {question}

    You must reason through the {keyword}s step by step and conclude with a final answer in this exact format:

    Answer: 'yes'

    Where:
    - Answer: 'yes' means that based on the {keyword}s the answer to the question is yes.
    - Answer: 'no' means that based on the {keyword}s the answer to the question is no.

    Use the format exactly and do not include additional text after your answer."""

    return sys_prompt

def get_role_of_var_to_optimize():
    role_description=(
        "A yes/no question designed for binary classification to minimize the weighted Gini score. "
        "The question should classify positive class samples by yielding 'yes' answers and negative class samples by yielding 'no' answers. "
        "The goal is to iteratively refine the question to improve class separation while maintaining correct classifications and fixing misclassifications."
    )
    return role_description

def get_analyze_group_prompt(
        current_question,
        predicted_label,
        well_classified,
        mismatched_label,
        misclassified,
        keyword,
        data_type,
    ):
    """Prompt used for getting feedback on the classification quality during training"""
    
    data_type = remove_article_from_data_type(data_type)

    prompt = f"""
    Below there are two groups of {data_type}s for which a model tried to predict their label (yes or no).
    Provide feedback about key {keyword}s that are present or absent in the group for which the model predicted the correct label.
    And provide feedback about key {keyword}s that are present or absent in the group for which the model predicted the wrong label.
    
    For these two groups of inputs, the model answered "{predicted_label}":
    * Well-classified {data_type}s (true label = "{predicted_label}"):
    {well_classified}
    * Misclassified {data_type}s (true label = "{mismatched_label}"):
    {misclassified}

    The feedback you provide must be clear and concise. Focus on the one or two most important {keyword}s and keep your answer short.
    """

    return prompt