import platformdirs
from .base import Dataset

class DIAGNO(Dataset):
    def __init__(self, subset:str, root: str=None, split: str="train", *args, **kwargs):
        """
        GSM8K dataset from HF."""
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        self.root = root
        self.subset = subset
        assert split in ["train", "val", "test"]
        if split == "test":
            self.data = load_dataset("ninaa510/diagnosis-text", split="test")
        elif split == "val":
            # Split the training set into half. Let the second half be the training set.
            # Let the first 100 samples be the validation set.
            self.data = load_dataset("ninaa510/diagnosis-text", split="test")
        elif split == "train":
            self.data = load_dataset("ninaa510/diagnosis-text", split="train")
        self.split = split
    
    def __getitem__(self, index):
        row = self.data[index]
        question = row["symptoms"]
        label = row["answer"]
        question_prompt = f"Symptoms: {question}"
        return question_prompt, label

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return "You will answer a diagnostic question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value where 1 means diagnosed with Tuberculosis and 0 means not diagnosed with Tuberculosis."
                
class DIAGNO_DSPy3(DIAGNO):
    def __init__(self, root: str = None, split: str = "train", seed: int = 42):
        """Balanced diagnosis dataset:
        - 300 training samples (150 per class)
        - 100 validation samples (50 per class)
        - Balanced test set
        """
        import tqdm
        import random
        from datasets import load_dataset
        from collections import Counter
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        
        dataset = load_dataset("ninaa510/diagnosis-text", 'default', cache_dir=root)
        train_data = dataset['train']
        test_data = dataset['test']

        def balance_data(dataset, n_per_class):
            pos, neg = [], []
            for example in dataset:
                label = example["label"].strip().lower()
                question = example["sentence1"]
                if label == "tuberculosis":
                    pos.append({"symptoms": question, "answer": 1})
                elif label == "allergic sinusitis":
                    neg.append({"symptoms": question, "answer": 0})
            rng = random.Random(seed)
            return rng.sample(pos, n_per_class) + rng.sample(neg, n_per_class)
        
        def full_dataset(dataset):
            data = []
            for example in dataset:
                label = example["label"].strip().lower()
                question = example["sentence1"]
                if label == "tuberculosis":
                    data.append({"symptoms": question, "answer": 1})
                elif label == "allergic sinusitis":
                    data.append({"symptoms": question, "answer": 0})
            return data

        # Create balanced datasets
        full_balanced_train = balance_data(train_data, 300)  # 150 per class -> 300
        balanced_val = balance_data(train_data, 50)          # 50 per class  -> 100
        balanced_test = balance_data(test_data, 300)         # 100 per class -> 200

        # Shuffle for randomness
        rng = random.Random(seed)
        rng.shuffle(full_balanced_train)
        rng.shuffle(balanced_val)
        rng.shuffle(balanced_test)

        if split == "train":
            self.data = full_balanced_train
        elif split == "val":
            self.data = balanced_val
        elif split == "test":
            self.data = balanced_test

    def __call__(self, x):
        return x['question']

    def get_task_description(self):
        return """You are a medical diagnostic assistant. Given a patient's symptom description, your job is to decide if the case indicates Tuberculosis.

    You must reason through the symptoms step by step and conclude with a final diagnosis in this exact format:

    Answer: 1

    Where:
    - Answer: 1 means the patient is diagnosed with Tuberculosis
    - Answer: 0 means the patient is not diagnosed with Tuberculosis

    Use the format exactly and do not include additional text after your answer.
    """

class DIAGNO_DSPy3_FullTest(DIAGNO_DSPy3):
    """Same as DIAGNO_DSPy3 but uses full unbalanced test set"""
    def __init__(self, root: str = None, split: str = "train", seed: int = 42):
        import random
        from datasets import load_dataset
        
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        
        dataset = load_dataset("ninaa510/diagnosis-text", 'default', cache_dir=root)
        train_data = dataset['train']
        test_data = dataset['test']

        def balance_data(dataset, n_per_class):
            pos, neg = [], []
            for example in dataset:
                label = example["label"].strip().lower()
                question = example["sentence1"]
                if label == "tuberculosis":
                    pos.append({"symptoms": question, "answer": 1})
                elif label == "allergic sinusitis":
                    neg.append({"symptoms": question, "answer": 0})
            rng = random.Random(seed)
            return rng.sample(pos, n_per_class) + rng.sample(neg, n_per_class)
        
        def full_dataset(dataset):
            """Get all data without balancing"""
            data = []
            for example in dataset:
                label = example["label"].strip().lower()
                question = example["sentence1"]
                if label == "tuberculosis":
                    data.append({"symptoms": question, "answer": 1})
                elif label == "allergic sinusitis":
                    data.append({"symptoms": question, "answer": 0})
            return data

        # For train and val, use balanced data
        full_balanced_train = balance_data(train_data, 300)
        balanced_val = balance_data(train_data, 50)
        
        # For test, use FULL unbalanced data
        full_test = full_dataset(test_data)

        rng = random.Random(seed)
        rng.shuffle(full_balanced_train)
        rng.shuffle(balanced_val)
        rng.shuffle(full_test)

        if split == "train":
            self.data = full_balanced_train
        elif split == "val":
            self.data = balanced_val
        elif split == "test":
            self.data = full_test  # FULL TEST SET

def structured_eval_fn(prediction_text):
    """Validate that the response follows the required structure"""
    try:
        # Check for 3 scores between 1-5
        scores = re.findall(r"Score: ([1-5])", prediction_text)
        if len(scores) != 3:
            return 0
        
        # Check for final diagnosis of 0 or 1
        final = re.findall(r"Final diagnosis: ([01])", prediction_text)
        if len(final) != 1:
            return 0
            
        return 1
    except:
        return 0

class DIAGNO_FULL(DIAGNO_DSPy3):
    """
    DIAGNO_FULL:
    - test split: FULL unbalanced test set (same as DIAGNO_DSPy3_FullTest behavior)
    - train split: as large as possible while matching the (0/1) label distribution of the FULL test set
      (i.e., pick the largest subset of train with the same ratio as test).
    - val split: kept the same as DIAGNO_DSPy3 (balanced) unless you want otherwise.
    """
    def __init__(self, root: str = None, split: str = "train", seed: int = 42):
        import random
        import math
        from math import gcd
        from datasets import load_dataset

        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        dataset = load_dataset("ninaa510/diagnosis-text", "default", cache_dir=root)
        train_data = dataset["train"]
        test_data  = dataset["test"]

        def to_binary_dataset(hf_split):
            """Convert HF split to list[dict] with keys: symptoms, answer (0/1)."""
            out = []
            for ex in hf_split:
                label = ex["label"].strip().lower()
                question = ex["sentence1"]
                if label == "tuberculosis":
                    out.append({"symptoms": question, "answer": 1})
                elif label == "allergic sinusitis":
                    out.append({"symptoms": question, "answer": 0})
            return out

        # FULL (unbalanced) test set
        full_test = to_binary_dataset(test_data)

        # All train examples (unbalanced)
        full_train = to_binary_dataset(train_data)

        # Split train by class
        train_0 = [ex for ex in full_train if ex["answer"] == 0]
        train_1 = [ex for ex in full_train if ex["answer"] == 1]

        # Compute test distribution counts
        test_0 = sum(1 for ex in full_test if ex["answer"] == 0)
        test_1 = sum(1 for ex in full_test if ex["answer"] == 1)

        if test_0 == 0 or test_1 == 0:
            raise ValueError(
                f"DIAGNO_FULL requires both classes to exist in test. Got test_0={test_0}, test_1={test_1}."
            )

        # Reduce the desired ratio to smallest integers (a:b) for exact matching
        g = gcd(test_0, test_1)
        a = test_0 // g  # class 0 units
        b = test_1 // g  # class 1 units

        # Maximize k so that k*a <= available_0 and k*b <= available_1
        avail_0 = len(train_0)
        avail_1 = len(train_1)
        k = min(avail_0 // a, avail_1 // b)

        if k == 0:
            raise ValueError(
                f"Not enough train data to match test distribution exactly. "
                f"Need at least a={a} zeros and b={b} ones, but have avail_0={avail_0}, avail_1={avail_1}."
            )

        n0 = k * a
        n1 = k * b

        rng = random.Random(seed)
        rng.shuffle(train_0)
        rng.shuffle(train_1)

        matched_train = train_0[:n0] + train_1[:n1]
        rng.shuffle(matched_train)

        # Keep val behavior identical to DIAGNO_DSPy3 (balanced val = 50 per class)
        def balance_data(hf_split, n_per_class):
            pos, neg = [], []
            for ex in hf_split:
                label = ex["label"].strip().lower()
                question = ex["sentence1"]
                if label == "tuberculosis":
                    pos.append({"symptoms": question, "answer": 1})
                elif label == "allergic sinusitis":
                    neg.append({"symptoms": question, "answer": 0})
            rng_local = random.Random(seed)
            return rng_local.sample(pos, n_per_class) + rng_local.sample(neg, n_per_class)

        balanced_val = balance_data(train_data, 50)  # 50 per class -> 100
        rng.shuffle(balanced_val)

        if split == "train":
            self.data = matched_train
        elif split == "val":
            self.data = balanced_val
        elif split == "test":
            self.data = full_test
        else:
            raise ValueError("split must be one of ['train','val','test']")