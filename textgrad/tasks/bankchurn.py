import platformdirs
from .base import Dataset
import json
import random
import os
import json

# Possible paths for .json file
possible_paths = [
    # Add path to bankchurn dataset created from:
    # /Users/timarni/Documents/ACT/textgrad/tasks/bankchurn_dataset_creation.ipynb
]

class BANKCHURN(Dataset):
    def __init__(self, subset: str, root: str = None, split: str = "train", *args, **kwargs):
        """Bank Customer Churn prediction dataset."""
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        
        self.root = root
        self.subset = subset
        assert split in ["train", "val", "test"]
        
        # Load JSON data from file
        json_path = None
        for path in possible_paths:
            if os.path.exists(path):
                json_path = path
                break

        if json_path is None:
            raise FileNotFoundError(f"Could not find bank_churn_dataset.json in any of these locations: {possible_paths}")

        with open(json_path, "r") as f:  
            all_data = json.load(f)
        
        # Filter by split field in JSON
        if split == "train":
            self.data = [item for item in all_data if item["split"] == "train"]
        elif split == "val":
            self.data = [item for item in all_data if item["split"] == "train"]
        elif split == "test":
            self.data = [item for item in all_data if item["split"] == "test"]
        
        self.split = split
    
    def __getitem__(self, index):
        row = self.data[index]
        text = row["text"]
        label = row["class"]  # 0 for no churn, 1 for churn
        
        question_prompt = f"Customer profile: {text}"
        return question_prompt, label
    
    def __len__(self):
        return len(self.data)
    
    def get_task_description(self):
        return "You will predict whether a bank customer will churn (leave the bank). Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is 0 if the customer will not churn or 1 if the customer will churn."

class BANKCHURN_DSPy(Dataset):
    def __init__(self, root: str = None, split: str = "train", seed: int = 42):
        """Balanced bank churn dataset for DSPy - 600 samples per split."""
        
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        
        # Load JSON data
        json_path = None
        for path in possible_paths:
            if os.path.exists(path):
                json_path = path
                break

        if json_path is None:
            raise FileNotFoundError(f"Could not find bank_churn_dataset.json in any of these locations: {possible_paths}")

        with open(json_path, "r") as f:
            all_data = json.load(f)
        
        # Separate by split field
        train_data = [item for item in all_data if item["split"] == "train"]
        test_data = [item for item in all_data if item["split"] == "test"]
        
        # Choose dataset based on split
        source_data = train_data if split == "train" else test_data
        
        # Separate by class
        no_churn = [item for item in source_data if item["class"] == 0]
        churn = [item for item in source_data if item["class"] == 1]
        
        # Balance the dataset - 300 samples per class = 600 total, load 350 for val set
        samples_per_class = 300
        
        rng = random.Random(seed)
        balanced_no_churn = rng.sample(no_churn, min(samples_per_class, len(no_churn)))
        balanced_churn = rng.sample(churn, min(samples_per_class, len(churn)))
        
        # Combine and shuffle
        balanced_data = []
        for item in balanced_no_churn:
            balanced_data.append({
                "text": item["text"],
                "answer": 0  # no churn
            })
        for item in balanced_churn:
            balanced_data.append({
                "text": item["text"], 
                "answer": 1  # churn
            })
        
        rng.shuffle(balanced_data)
        
        if split == "train":
            self.data = balanced_data[:2*samples_per_class]
        elif split == "val":
            self.data = balanced_data[:100]  # Small val set
        elif split == "test":
            self.data = balanced_data[:2*samples_per_class]
    
    def __getitem__(self, index):
        row = self.data[index]
        text = row["text"]
        answer = row["answer"]
        
        question_prompt = f"Customer profile: {text}"
        return question_prompt, answer
    
    def __len__(self):
        return len(self.data)
    
    def get_task_description(self):
        return """You are a customer churn prediction assistant. Given a bank customer's profile, determine if they will churn (leave the bank).

Analyze the customer characteristics including credit score, account balance, tenure, product usage, and activity level. Conclude with your prediction in this exact format:

Answer: 0

Where:
- Answer: 0 means the customer will NOT churn (will stay with the bank)
- Answer: 1 means the customer WILL churn (will leave the bank)

Use the format exactly and do not include additional text after your answer."""

class BANKCHURN_DSPy_FullTest(BANKCHURN_DSPy):
    """Same as BANKCHURN_DSPy but uses full unbalanced test set"""
    def __init__(self, root: str = None, split: str = "train", seed: int = 42):
        
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        
        # Load JSON data
        json_path = None
        for path in possible_paths:
            if os.path.exists(path):
                json_path = path
                break

        if json_path is None:
            raise FileNotFoundError(f"Could not find bank_churn_dataset.json in any of these locations: {possible_paths}")

        with open(json_path, "r") as f:
            all_data = json.load(f)
        
        # Separate by split field
        train_data = [item for item in all_data if item["split"] == "train"]
        test_data = [item for item in all_data if item["split"] == "test"]
        
        def balance_data(source_data, samples_per_class):
            """Balance the dataset"""
            no_churn = [item for item in source_data if item["class"] == 0]
            churn = [item for item in source_data if item["class"] == 1]
            
            rng = random.Random(seed)
            balanced_no_churn = rng.sample(no_churn, min(samples_per_class, len(no_churn)))
            balanced_churn = rng.sample(churn, min(samples_per_class, len(churn)))
            
            balanced_data = []
            for item in balanced_no_churn:
                balanced_data.append({"text": item["text"], "answer": 0})
            for item in balanced_churn:
                balanced_data.append({"text": item["text"], "answer": 1})
            
            rng.shuffle(balanced_data)
            return balanced_data
        
        def full_dataset(source_data):
            """Get all data without balancing"""
            data = []
            for item in source_data:
                data.append({
                    "text": item["text"],
                    "answer": item["class"]
                })
            rng = random.Random(seed)
            rng.shuffle(data)
            return data
        
        # For train and val, use balanced data
        if split == "train":
            balanced_train = balance_data(train_data, 300)
            self.data = balanced_train[:600]  # 300 per class
        elif split == "val":
            balanced_train = balance_data(train_data, 300)
            self.data = balanced_train[:100]  # Small val set
        elif split == "test":
            # For test, use FULL unbalanced data
            self.data = full_dataset(test_data)


class BANKCHURN_DSPy_Imbalanced(Dataset):
    """Bank churn dataset with original imbalanced distribution."""
    def __init__(self, root: str = None, split: str = "train", seed: int = 42, train_size: int = 600, val_size: int = 300):
        
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        
        # Load JSON data
        json_path = None
        for path in possible_paths:
            if os.path.exists(path):
                json_path = path
                break

        if json_path is None:
            raise FileNotFoundError(f"Could not find bank_churn_dataset.json in any of these locations: {possible_paths}")

        with open(json_path, "r") as f:
            all_data = json.load(f)
        
        # Separate by split field
        train_data = [item for item in all_data if item["split"] == "train"]
        test_data = [item for item in all_data if item["split"] == "test"]
        
        rng = random.Random(seed)
        
        if split in ["train", "val"]:
            # Separate by class
            no_churn_all = [item for item in train_data if item["class"] == 0]
            churn_all = [item for item in train_data if item["class"] == 1]
            
            # Shuffle each class
            rng.shuffle(no_churn_all)
            rng.shuffle(churn_all)
            
            # Calculate original distribution
            total = len(train_data)
            no_churn_ratio = len(no_churn_all) / total
            
            # Calculate sizes for train+val combined, maintaining distribution
            combined_size = train_size + val_size
            combined_no_churn_size = int(combined_size * no_churn_ratio)
            combined_churn_size = combined_size - combined_no_churn_size
            
            # Sample from each class for combined pool
            sampled_no_churn = no_churn_all[:combined_no_churn_size]
            sampled_churn = churn_all[:combined_churn_size]
            
            # Now split into val and train, maintaining distribution
            val_no_churn_size = int(val_size * no_churn_ratio)
            val_churn_size = val_size - val_no_churn_size
            
            val_no_churn = sampled_no_churn[:val_no_churn_size]
            val_churn = sampled_churn[:val_churn_size]
            
            train_no_churn = sampled_no_churn[val_no_churn_size:]
            train_churn = sampled_churn[val_churn_size:]
            
            # Convert to our format
            def convert_items(items):
                return [{"text": item["text"], "answer": item["class"]} for item in items]
            
            if split == "val":
                self.data = convert_items(val_no_churn) + convert_items(val_churn)
            else:  # train
                self.data = convert_items(train_no_churn) + convert_items(train_churn)
            
            rng.shuffle(self.data)
                
        elif split == "test":
            # Use full test set as-is
            self.data = []
            for item in test_data:
                self.data.append({
                    "text": item["text"],
                    "answer": item["class"]
                })
            rng.shuffle(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        text = row["text"]
        answer = row["answer"]
        
        question_prompt = f"Customer profile: {text}"
        return question_prompt, answer
    
    def __len__(self):
        return len(self.data)
    
    def get_task_description(self):
        return """You are a customer churn prediction assistant. Given a bank customer's profile, determine if they will churn (leave the bank).

Analyze the customer characteristics including credit score, account balance, tenure, product usage, and activity level. Conclude with your prediction in this exact format:

Answer: 0

Where:
- Answer: 0 means the customer will NOT churn (will stay with the bank)
- Answer: 1 means the customer WILL churn (will leave the bank)

Use the format exactly and do not include additional text after your answer."""


class BANKCHURN_DSPy_Full(Dataset):
    """Bank churn dataset with original imbalanced distribution."""
    def __init__(self, root: str = None, split: str = "train", seed: int = 42, val_size: int = 300):
        
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        
        # Load JSON data
        json_path = None
        for path in possible_paths:
            if os.path.exists(path):
                json_path = path
                break

        if json_path is None:
            raise FileNotFoundError(f"Could not find bank_churn_dataset.json in any of these locations: {possible_paths}")

        with open(json_path, "r") as f:
            all_data = json.load(f)
        
        # Separate by split field
        train_data = [item for item in all_data if item["split"] == "train"]
        test_data = [item for item in all_data if item["split"] == "test"]
        
        rng = random.Random(seed)
        
        if split in ["train", "val"]:
            # Convert train data to our format
            all_train = []
            for item in train_data:
                all_train.append({
                    "text": item["text"],
                    "answer": item["class"]
                })
            
            # Shuffle with seed for reproducibility
            rng.shuffle(all_train)
            
            # Calculate class distribution for stratified val sampling
            no_churn_train = [item for item in all_train if item["answer"] == 0]
            churn_train = [item for item in all_train if item["answer"] == 1]
            
            # Calculate proportions
            total = len(all_train)
            no_churn_ratio = len(no_churn_train) / total
            churn_ratio = len(churn_train) / total
            
            # Stratified sampling for val set
            val_no_churn_size = int(val_size * no_churn_ratio)
            val_churn_size = val_size - val_no_churn_size  # Ensure exactly val_size samples
            
            # Sample val set from each class
            val_no_churn = no_churn_train[:val_no_churn_size]
            val_churn = churn_train[:val_churn_size]
            
            # Remaining goes to train
            train_no_churn = no_churn_train[val_no_churn_size:]
            train_churn = churn_train[val_churn_size:]
            
            if split == "val":
                self.data = val_no_churn + val_churn
                rng.shuffle(self.data)
            else:  # train
                self.data = train_no_churn + train_churn
                rng.shuffle(self.data)
                
        elif split == "test":
            # Use full test set as-is
            self.data = []
            for item in test_data:
                self.data.append({
                    "text": item["text"],
                    "answer": item["class"]
                })
            rng.shuffle(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        text = row["text"]
        answer = row["answer"]
        
        question_prompt = f"Customer profile: {text}"
        return question_prompt, answer
    
    def __len__(self):
        return len(self.data)
    
    def get_task_description(self):
        return """You are a customer churn prediction assistant. Given a bank customer's profile, determine if they will churn (leave the bank).

Analyze the customer characteristics including credit score, account balance, tenure, product usage, and activity level. Conclude with your prediction in this exact format:

Answer: 0

Where:
- Answer: 0 means the customer will NOT churn (will stay with the bank)
- Answer: 1 means the customer WILL churn (will leave the bank)

Use the format exactly and do not include additional text after your answer."""