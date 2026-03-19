import platformdirs
from .base import Dataset

class SPAM(Dataset):
    def __init__(self, subset:str, root: str=None, split: str="train", *args, **kwargs):
        """
        spam dataset from HF."""
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        self.root = root
        self.subset = subset
        assert split in ["train", "val", "test"]
        if split == "test":
            self.data = load_dataset("Deysi/spam-detection-dataset", split="test")
        elif split == "val":
            # Split the training set into half. Let the second half be the training set.
            # Let the first 100 samples be the validation set.
            self.data = load_dataset("Deysi/spam-detection-dataset", split="test")
        elif split == "train":
            self.data = load_dataset("Deysi/spam-detection-dataset", split="train")
        self.split = split
    
    def __getitem__(self, index):
        row = self.data[index]
        question = row["text"]
        label = row["label"]
        #print(label)
        
        #answer = 1 if label == "Tuberculosis" else 0
        #answer = 1 if label.lower() == "tuberculosis".lower() else 0
        question_prompt = f"Question: {question}"
        return question_prompt, label

    def __len__(self):
        return len(self.data)

    #def get_task_description(self):
    #    return "You will answer a diagnostic question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value where 0 is diagnosed and 1 is non diagnosed."
    def get_task_description(self):
        return "You will answer a spam detection question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value where 1 means detected as spam and 0 means not detected as spam."

class SPAM_DSPy3(SPAM):
    def __init__(self, root: str = None, split: str = "train", seed: int = 42):
        """Balanced spam dataset:
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
        
        dataset = load_dataset("Deysi/spam-detection-dataset", 'default', cache_dir=root)
        train_data = dataset['train']
        test_data = dataset['test']

        def balance_data(dataset, n_per_class):
            pos, neg = [], []
            for example in dataset:
                label = example["label"].strip().lower()
                question = example["text"]
                if label == "spam":
                    pos.append({"text": question, "label": 1})
                elif label == "not_spam":
                    neg.append({"text": question, "label": 0})
            rng = random.Random(seed)
            return rng.sample(pos, n_per_class) + rng.sample(neg, n_per_class)

        # Create balanced datasets
        full_balanced_train = balance_data(train_data, 300)  # 150 per class -> 300
        balanced_val = balance_data(train_data, 100)          # 50 per class  -> 100
        balanced_test = balance_data(test_data, 1300)         # 100 per class -> 200

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
        return """You are a spam detection assistant. Given the body of an email, your job is to decide if the email is a spam.

    You must reason through the content and form of the email step by step, and conclude with a final decision in this exact format:

    Answer: 1

    Where:
    - Answer: 1 means the email is detected as spam
    - Answer: 0 means the email is not detected as spam

    Use the format exactly and do not include additional text after your answer.
    """

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
    
class SPAM_DSPy3_FullTest(SPAM_DSPy3):
    """Same as SPAM_DSPy3 but uses full unbalanced test set"""
    def __init__(self, root: str = None, split: str = "train", seed: int = 42):
        import random
        from datasets import load_dataset
        
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        
        dataset = load_dataset("Deysi/spam-detection-dataset", 'default', cache_dir=root)
        train_data = dataset['train']
        test_data = dataset['test']

        def balance_data(dataset, n_per_class):
            pos, neg = [], []
            for example in dataset:
                label = example["label"].strip().lower()
                question = example["text"]
                if label == "spam":
                    pos.append({"text": question, "label": 1})
                elif label == "not_spam":
                    neg.append({"text": question, "label": 0})
            rng = random.Random(seed)
            return rng.sample(pos, n_per_class) + rng.sample(neg, n_per_class)
        
        def full_dataset(dataset):
            """Get all data without balancing"""
            data = []
            for example in dataset:
                label = example["label"].strip().lower()
                question = example["text"]
                if label == "spam":
                    data.append({"text": question, "label": 1})
                elif label == "not_spam":
                    data.append({"text": question, "label": 0})
            return data

        # For train and val, use balanced data
        full_balanced_train = balance_data(train_data, 600)
        balanced_val = balance_data(train_data, 100)
        
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

class SPAM_FULL_DATASET(Dataset):
    """
    Loads the full HF train and full HF test.
    Creates a validation split from train using stratified sampling so the val
    distribution matches the train distribution (as closely as possible).
    """
    def __init__(self, root: str = None, split: str = "train", seed: int = 42, val_size=0.05):
        import random
        from datasets import load_dataset

        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        assert split in ["train", "val", "test"]
        rng = random.Random(seed)

        ds = load_dataset("Deysi/spam-detection-dataset", "default", cache_dir=root)
        hf_train = ds["train"]
        hf_test = ds["test"]

        def to_records(hf_split):
            out = []
            for ex in hf_split:
                lab = ex["label"].strip().lower()
                if lab == "spam":
                    y = 1
                elif lab == "not_spam":
                    y = 0
                else:
                    continue
                out.append({"text": ex["text"], "label": y})
            return out

        train_all = to_records(hf_train)
        test_all = to_records(hf_test)

        # --- stratified val split from train_all ---
        n = len(train_all)
        if isinstance(val_size, float):
            k = max(1, int(round(n * val_size)))
        else:
            k = max(1, min(int(val_size), n - 1))

        idx_1 = [i for i, r in enumerate(train_all) if r["label"] == 1]
        idx_0 = [i for i, r in enumerate(train_all) if r["label"] == 0]
        rng.shuffle(idx_1)
        rng.shuffle(idx_0)

        # match train distribution
        p1 = len(idx_1) / n if n else 0.0
        k1 = int(round(k * p1))
        k1 = min(k1, len(idx_1))
        k0 = k - k1
        if k0 > len(idx_0):
            k0 = len(idx_0)
            k1 = min(k - k0, len(idx_1))

        val_idx = set(idx_1[:k1] + idx_0[:k0])
        train_idx = [i for i in range(n) if i not in val_idx]

        train_split = [train_all[i] for i in train_idx]
        val_split = [train_all[i] for i in val_idx]

        rng.shuffle(train_split)
        rng.shuffle(val_split)
        rng.shuffle(test_all)

        if split == "train":
            self.data = train_split
        elif split == "val":
            self.data = val_split
        else:
            self.data = test_all

    def __getitem__(self, index):
        row = self.data[index]
        question_prompt = f"Question: {row['text']}"
        return question_prompt, row["label"]

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return """You are a spam detection assistant. Given the body of an email, your job is to decide if the email is a spam.

    You must reason through the content and form of the email step by step, and conclude with a final decision in this exact format:

    Answer: 1

    Where:
    - Answer: 1 means the email is detected as spam
    - Answer: 0 means the email is not detected as spam

    Use the format exactly and do not include additional text after your answer.
    """