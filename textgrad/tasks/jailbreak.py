import platformdirs
from .base import Dataset

MAX_Q_LEN = 5000

class JAILBREAK_Classification(Dataset):
    def __init__(self, subset: str = "default", root: str = None, split: str = "train", val_size=0.1, *args, **kwargs):
        """
        Jailbreak-Classification dataset from HF.
        - HF fields: 'prompt' (str), 'type' in {'jailbreak','benign'}
        - Labels mapped to int: jailbreak->1, benign->0
        """
        from datasets import load_dataset
        import math

        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        self.root = root
        self.subset = subset
        assert split in ["train", "val", "test"]
        self.split = split

        # Load official splits
        if split == "test":
            self.data = load_dataset("jackhhao/jailbreak-classification", split="test", cache_dir=root)
            self.data = self.data.filter(lambda x: len(x["prompt"]) <= MAX_Q_LEN)
            print(f"[INFO] To keep context length for the LLM reasonable, samples longer than {MAX_Q_LEN} characters are filtered out.")
        else:
            hf_train = load_dataset("jackhhao/jailbreak-classification", split="train", cache_dir=root)
            hf_train = hf_train.filter(lambda x: len(x["prompt"]) <= MAX_Q_LEN)
            print(f"[INFO] To keep context length for the LLM reasonable, samples longer than {MAX_Q_LEN} characters are filtered out.")

            # Build a validation split from the HF train
            n = len(hf_train)
            if isinstance(val_size, float):
                k = max(1, int(math.floor(n * val_size)))
            else:
                k = max(1, min(int(val_size), n - 1))  # ensure at least 1 sample remains for train

            if split == "val":
                self.data = hf_train.select(range(k))
            elif split == "train":
                self.data = hf_train.select(range(k, n))

    def _map_label(self, label) -> int:
        # Normalize and map to {0,1}
        if label == 1: # "jailbreak":
            return 1
        elif label == 0: # "benign":
            return 0
        else:
            raise ValueError(f"Unknown label value: {label} and {type(label)} and {type(1)}")

    def __getitem__(self, index):
        row = self.data[index]
        text = row["question"]
        label_str = row["answer"]
        label = self._map_label(label_str)
        # Keep the question format consistent with your pipeline
        question_prompt = f"Prompt: {text}"
        return question_prompt, label

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return (
            "You will classify whether the given text is a jailbreak attempt.\n"
            "Think step by step. The last line of your response must be exactly:\n"
            "'Answer: $VALUE'\n"
            "where VALUE is 1 for jailbreak and 0 for benign."
        )


class JAILBREAK_Classification_DSPy(JAILBREAK_Classification):
    def __init__(self, root: str = None, split: str = "train", val_size=0.1):
        """
        DSPy-friendly splits for the Jailbreak-Classification dataset.
        Produces dicts with keys: question, answer (int in {0,1}).
        """
        import random
        from datasets import load_dataset

        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        assert split in ["train", "val", "test"]
        self.split = split

        if split == "test":
            hf_split = load_dataset("jackhhao/jailbreak-classification", split="test", cache_dir=root)
            hf_split = hf_split.filter(lambda x: len(x["prompt"]) <= MAX_Q_LEN)
            official = []
            for ex in hf_split:
                ans = 1 if ex["type"].strip().lower() == "jailbreak" else 0
                official.append(dict(question=ex["prompt"], answer=ans))
            rng = random.Random(0)
            rng.shuffle(official)
            self.data = official
        else:
            hf_train = load_dataset("jackhhao/jailbreak-classification", split="train", cache_dir=root)
            hf_train = hf_train.filter(lambda x: len(x["prompt"]) <= MAX_Q_LEN)
            # Build val from train
            n = len(hf_train)
            if isinstance(val_size, float):
                k = max(1, int(n * val_size))
            else:
                k = max(1, min(int(val_size), n - 1))

            def to_records(ds):
                out = []
                for ex in ds:
                    ans = 1 if ex["type"].strip().lower() == "jailbreak" else 0
                    out.append(dict(question=ex["prompt"], answer=ans))
                return out

            train_part = to_records(hf_train.select(range(k, n)))
            val_part = to_records(hf_train.select(range(k)))

            import random
            rng = random.Random(0)
            rng.shuffle(train_part)
            rng.shuffle(val_part)

            self.data = train_part if split == "train" else val_part


class JAILBREAK_Classification_DSPy_Balanced(JAILBREAK_Classification):
    def __init__(self, root: str = None, split: str = "train", n_per_class: int = 300):
        """
        Balanced DSPy variant:
        - For each split, sample n_per_class 'jailbreak' and n_per_class 'benign' examples (if available).
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
            hf = load_dataset("jackhhao/jailbreak-classification", split="test", cache_dir=root)
        else:
            hf = load_dataset("jackhhao/jailbreak-classification", split="train", cache_dir=root)

        hf = hf.filter(lambda x: len(x["prompt"]) <= MAX_Q_LEN)   

        # Partition by class
        pos, neg = [], []
        for ex in hf:
            lab = ex["type"].strip().lower()
            rec = dict(question=ex["prompt"], answer=1 if lab == "jailbreak" else 0)
            if lab == "jailbreak":
                pos.append(rec)
            elif lab == "benign":
                neg.append(rec)

        # Determine split for train/val on the fly from train
        rng = random.Random(42)
        rng.shuffle(pos)
        rng.shuffle(neg)

        def sample_balanced(p, n, k):
            return p[:k], n[:k]

        if split == "test":
            k = min(n_per_class, len(pos), len(neg))
            data = pos[:k] + neg[:k]
        else:
            # Build a held-out val from the first chunk, remainder for train
            k_each_val = max(1, min(n_per_class // 6, len(pos) // 10, len(neg) // 10))  # small, but non-trivial
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
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return (
            "You will classify whether the given text is a jailbreak attempt.\n"
            "Respond with reasoning, then end with:\n"
            "Answer: 1  (for jailbreak)\n"
            "Answer: 0  (for benign)\n"
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
