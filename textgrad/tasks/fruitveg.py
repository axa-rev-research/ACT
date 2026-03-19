import platformdirs
from .base import Dataset
import json
import random

class FRUITVEG(Dataset):
    def __init__(self, subset: str, root: str = None, split: str = "train", *args, **kwargs):
        """Fruit/Vegetable classification dataset."""
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        
        self.root = root
        self.subset = subset
        assert split in ["train", "val", "test"]
        
        # Load hardcoded JSON data from file     
        with open("/mloscratch/users/arni/Deep_Agents_Network/veg_data.json", "r") as f:
            all_data = json.load(f)
        
        # Split the data
        total_len = len(all_data) # TODO Modify this if it would be a real dataset
        
        if split == "train":
            self.data = all_data[:total_len]
        elif split == "val":
            self.data = all_data[:total_len]
        elif split == "test":
            self.data = all_data[:total_len]
        
        self.split = split
    
    def __getitem__(self, index):
        row = self.data[index]
        text = row["text"]
        label = row["class"]  # 0 for fruit, 1 for vegetable
        
        question_prompt = f"Example: {text}"
        return question_prompt, label
    
    def __len__(self):
        return len(self.data)
    
    def get_task_description(self):
        return "You will classify whether a description is about a fruit or a vegetable. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is 0 for fruit or 1 for vegetable."

class FRUITVEG_DSPy(Dataset):
    def __init__(self, root: str = None, split: str = "train", seed: int = 42):
        """Balanced fruit/vegetable dataset for DSPy."""
        import random
        import json
        
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        
        # Load your JSON data
        with open("/mloscratch/users/arni/Deep_Agents_Network/veg_data.json", "r") as f:
            all_data = json.load(f)
        
        # Separate fruits and vegetables
        fruits = [item for item in all_data if item["class"] == 0]
        vegetables = [item for item in all_data if item["class"] == 1]
        
        # Balance the dataset (equal numbers of each class)
        min_class_size = min(len(fruits), len(vegetables))
        
        # Sample equally from both classes
        rng = random.Random(seed)
        balanced_fruits = rng.sample(fruits, min_class_size)
        balanced_vegetables = rng.sample(vegetables, min_class_size)
        
        # Combine and shuffle
        balanced_data = []
        for item in balanced_fruits:
            balanced_data.append({
                "text": item["text"],
                "answer": 0  # fruit
            })
        for item in balanced_vegetables:
            balanced_data.append({
                "text": item["text"], 
                "answer": 1  # vegetable
            })
        
        rng.shuffle(balanced_data)
        
        # Split into train/val/test
        total_len = len(balanced_data)
        
        if split == "train":
            self.data = balanced_data[:total_len]
        elif split == "val":
            self.data = balanced_data[:total_len]
        elif split == "test":
            self.data = balanced_data[:total_len]
    
    def __getitem__(self, index):
        row = self.data[index]
        text = row["text"]
        answer = row["answer"]
        
        question_prompt = f"Example: {text}"
        return question_prompt, answer
    
    def __len__(self):
        return len(self.data)
    
    def get_task_description(self):
        return """You are a classification assistant. Given a description, determine if it describes a fruit or a vegetable.

Analyze the key characteristics mentioned in the text and conclude with your classification in this exact format:

Answer: 0

Where:
- Answer: 0 means the description is about a fruit
- Answer: 1 means the description is about a vegetable

Use the format exactly and do not include additional text after your answer."""