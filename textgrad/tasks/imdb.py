import platformdirs
from .base import Dataset

MAX_REVIEW_LEN = 5000

class IMDB_Classification(Dataset):
    def __init__(self, subset: str = "default", root: str = None, split: str = "train", *args, **kwargs):
        """
        IMDB movie review sentiment classification dataset from HF.
        - HF fields: 'text' (str), 'label' (int) where 0=negative, 1=positive
        - Labels are already integers: 0 for negative, 1 for positive
        """
        from datasets import load_dataset

        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        self.root = root
        self.subset = subset
        assert split in ["train", "val", "test"]
        self.split = split

        # Load official splits
        if split == "test":
            self.data = load_dataset("stanfordnlp/imdb", split="test", cache_dir=root)
            self.data = self.data.filter(lambda x: len(x["text"]) <= MAX_REVIEW_LEN)
        else:
            # IMDB doesn't have a validation split, so we create one from train
            import random
            hf_train = load_dataset("stanfordnlp/imdb", split="train", cache_dir=root)
            hf_train = hf_train.filter(lambda x: len(x["text"]) <= MAX_REVIEW_LEN)

            n = len(hf_train)
            val_size = max(1, int(n * 0.1))
            rng = random.Random(kwargs.get("seed", 42))  # or add seed param to this class if you prefer

            idx_pos = [i for i, ex in enumerate(hf_train) if ex["label"] == 1]
            idx_neg = [i for i, ex in enumerate(hf_train) if ex["label"] == 0]
            rng.shuffle(idx_pos)
            rng.shuffle(idx_neg)

            p_pos = len(idx_pos) / n if n else 0.0
            k_pos = int(round(val_size * p_pos))
            k_pos = min(k_pos, len(idx_pos))
            k_neg = val_size - k_pos
            if k_neg > len(idx_neg):
                k_neg = len(idx_neg)
                k_pos = min(val_size - k_neg, len(idx_pos))

            val_idx = idx_pos[:k_pos] + idx_neg[:k_neg]
            val_idx_set = set(val_idx)
            train_idx = [i for i in range(n) if i not in val_idx_set]
            rng.shuffle(val_idx)
            rng.shuffle(train_idx)

            if split == "val":
                self.data = hf_train.select(val_idx)
            elif split == "train":
                self.data = hf_train.select(train_idx)


    def _map_label(self, label) -> int:
        # Labels are already 0 (negative) and 1 (positive)
        if label in [0, 1]:
            return label
        else:
            raise ValueError(f"Unknown label value: {label}")

    def __getitem__(self, index):
        row = self.data[index]
        text = row["text"]
        label = self._map_label(row["label"])
        # Format the input
        question_prompt = f"Review: {text}"
        return question_prompt, label

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return (
            "You will classify whether the given movie review expresses a positive or negative sentiment.\n"
            "Think step by step. The last line of your response must be exactly:\n"
            "'Answer: $VALUE'\n"
            "where VALUE is 0 for negative sentiment and 1 for positive sentiment."
        )


class IMDB_Classification_DSPy(IMDB_Classification):
    def __init__(self, root: str = None, split: str = "train", seed: int = 42, val_size: float = 0.1):
        """
        DSPy-friendly splits for the IMDB sentiment classification dataset.
        Produces dicts with keys: question, answer (int in {0,1}).
        """
        import random
        from datasets import load_dataset

        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        assert split in ["train", "val", "test"]
        self.split = split

        if split == "test":
            hf_split = load_dataset("stanfordnlp/imdb", split="test", cache_dir=root)
            hf_split = hf_split.filter(lambda x: len(x["text"]) <= MAX_REVIEW_LEN)
            official = []
            for ex in hf_split:
                official.append(dict(question=ex["text"], answer=ex["label"]))
            rng = random.Random(seed)
            rng.shuffle(official)
            self.data = official

        else:
            hf_train = load_dataset("stanfordnlp/imdb", split="train", cache_dir=root)
            hf_train = hf_train.filter(lambda x: len(x["text"]) <= MAX_REVIEW_LEN)

            n = len(hf_train)
            if isinstance(val_size, float):
                k = max(1, int(round(n * val_size)))
            else:
                k = max(1, min(int(val_size), n - 1))

            rng = random.Random(seed)

            idx_pos = [i for i, ex in enumerate(hf_train) if ex["label"] == 1]
            idx_neg = [i for i, ex in enumerate(hf_train) if ex["label"] == 0]
            rng.shuffle(idx_pos)
            rng.shuffle(idx_neg)

            p_pos = len(idx_pos) / n if n else 0.0
            k_pos = int(round(k * p_pos))
            k_pos = min(k_pos, len(idx_pos))
            k_neg = k - k_pos
            if k_neg > len(idx_neg):
                k_neg = len(idx_neg)
                k_pos = min(k - k_neg, len(idx_pos))

            val_idx = idx_pos[:k_pos] + idx_neg[:k_neg]
            val_idx_set = set(val_idx)
            train_idx = [i for i in range(n) if i not in val_idx_set]
            rng.shuffle(val_idx)
            rng.shuffle(train_idx)

            def to_records(indices):
                out = []
                for i in indices:
                    ex = hf_train[int(i)]
                    out.append(dict(question=ex["text"], answer=ex["label"]))
                return out

            train_part = to_records(train_idx)
            val_part = to_records(val_idx)

            self.data = train_part if split == "train" else val_part

    def __getitem__(self, index):
        item = self.data[index]
        question_prompt = f"Review: {item['question']}"
        return question_prompt, item['answer']

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return (
            "You will classify whether the given movie review expresses a positive or negative sentiment.\n"
            "Respond with reasoning, then end with:\n"
            "Answer: 0  (for negative sentiment)\n"
            "Answer: 1  (for positive sentiment)\n"
        )


class IMDB_Classification_DSPy_Balanced(IMDB_Classification):
    def __init__(self, root: str = None, split: str = "train", seed: int = 42, n_per_class: int = 600):
        """
        Balanced DSPy variant:
        - For each split, sample n_per_class 'positive' and n_per_class 'negative' examples (if available).
        - Returns dicts {question, answer}.
        """
        import random
        from datasets import load_dataset

        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        assert split in ["train", "val", "test"]
        self.split = split

        # Load split(s)
        if split == "test":
            hf = load_dataset("stanfordnlp/imdb", split="test", cache_dir=root)
        else:
            hf = load_dataset("stanfordnlp/imdb", split="train", cache_dir=root)

        hf = hf.filter(lambda x: len(x["text"]) <= MAX_REVIEW_LEN)   

        # Partition by class
        pos, neg = [], []
        for ex in hf:
            rec = dict(question=ex["text"], answer=ex["label"])
            if ex["label"] == 1:  # positive
                pos.append(rec)
            elif ex["label"] == 0:  # negative
                neg.append(rec)

        # Determine split for train/val on the fly from train
        rng = random.Random(seed)
        rng.shuffle(pos)
        rng.shuffle(neg)

        def sample_balanced(p, n, k):
            return p[:k], n[:k]

        if split == "test":
            k = min(n_per_class, len(pos), len(neg))
            data = pos[:k] + neg[:k]
        else:
            # Build a held-out val from the first chunk, remainder for train
            k_each_val = max(1, min(n_per_class // 6, len(pos) // 10, len(neg) // 10))
            val_pos, val_neg = sample_balanced(pos, neg, k_each_val)
            train_pos, train_neg = pos[k_each_val:], neg[k_each_val:]

            k_each_train = min(n_per_class, len(train_pos), len(train_neg))
            if split == "val":
                data = val_pos + val_neg
            else:
                data = train_pos[:k_each_train] + train_neg[:k_each_train]

        rng.shuffle(data)
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        question_prompt = f"Review: {item['question']}"
        return question_prompt, item['answer']

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return (
            "You will classify whether the given movie review expresses a positive or negative sentiment.\n"
            "Respond with reasoning, then end with:\n"
            "Answer: 0  (for negative sentiment)\n"
            "Answer: 1  (for positive sentiment)\n"
        )


def structured_eval_fn(prediction_text: str) -> int:
    """
    Validate that the response ends with 'Answer: 0' or 'Answer: 1'.
    Returns 1 if well-formed, else 0.
    """
    import re
    try:
        m = re.search(r"Answer:\s*([01])\s*$", prediction_text.strip())
        return 1 if m else 0
    except Exception:
        return 0

class IMDB_Classification_DSPy_Balanced_FullTest(IMDB_Classification_DSPy_Balanced):
    """Same as IMDB_Classification_DSPy_Balanced but uses full unbalanced test set"""
    def __init__(self, root: str = None, split: str = "train", seed: int = 42, n_per_class: int = 600):
        import random
        from datasets import load_dataset

        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        assert split in ["train", "val", "test"]
        self.split = split

        # Load datasets
        if split == "test":
            hf = load_dataset("stanfordnlp/imdb", split="test", cache_dir=root)
        else:
            hf = load_dataset("stanfordnlp/imdb", split="train", cache_dir=root)

        hf = hf.filter(lambda x: len(x["text"]) <= MAX_REVIEW_LEN)

        def balance_data(dataset, k):
            """Balance dataset to k samples per class"""
            pos, neg = [], []
            for ex in dataset:
                rec = dict(question=ex["text"], answer=ex["label"])
                if ex["label"] == 1:
                    pos.append(rec)
                elif ex["label"] == 0:
                    neg.append(rec)
            
            rng = random.Random(seed)
            rng.shuffle(pos)
            rng.shuffle(neg)
            
            k_actual = min(k, len(pos), len(neg))
            return pos[:k_actual] + neg[:k_actual]
        
        def full_dataset(dataset):
            """Get all data without balancing"""
            data = []
            for ex in dataset:
                data.append(dict(question=ex["text"], answer=ex["label"]))
            rng = random.Random(seed)
            rng.shuffle(data)
            return data

        # For train and val, use balanced data
        if split in ["train", "val"]:
            # Partition by class for train/val split
            pos, neg = [], []
            for ex in hf:
                rec = dict(question=ex["text"], answer=ex["label"])
                if ex["label"] == 1:
                    pos.append(rec)
                elif ex["label"] == 0:
                    neg.append(rec)

            rng = random.Random(seed)
            rng.shuffle(pos)
            rng.shuffle(neg)

            # Split into val and train
            k_each_val = max(1, min(n_per_class // 6, len(pos) // 10, len(neg) // 10))
            val_pos, val_neg = pos[:k_each_val], neg[:k_each_val]
            train_pos, train_neg = pos[k_each_val:], neg[k_each_val:]

            if split == "val":
                data = val_pos + val_neg
            else:
                k_each_train = min(n_per_class, len(train_pos), len(train_neg))
                data = train_pos[:k_each_train] + train_neg[:k_each_train]
            
            rng.shuffle(data)
            self.data = data
        
        elif split == "test":
            # For test, use FULL unbalanced data
            self.data = full_dataset(hf)