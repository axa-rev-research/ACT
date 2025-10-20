import platformdirs
from .base import Dataset

class DIAGNO(Dataset):
    def __init__(self, subset:str, root: str=None, split: str="train", *args, **kwargs):
        """DIAGNO dataset from HF."""
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        self.root = root
        self.subset = subset
        assert split in ["train", "val", "test"]
        if split == "test":
            self.data = load_dataset("ninaa510/diagnosis-text", split="test")
        elif split == "val":
            self.data = load_dataset("ninaa510/diagnosis-text", split="train")
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
    
class DIAGNO_DSPy(DIAGNO):
    def __init__(self, root:str=None, split: str="train"):
        """DSPy splits for the diagnosis dataset."""
        import tqdm
        import random
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        dataset = load_dataset("ninaa510/diagnosis-text", 'default', cache_dir=root)
        hf_official_train = dataset['train']
        hf_official_test = dataset['test']
        official_train = []
        official_test = []
        for example in tqdm.tqdm(hf_official_train):
            question = example['sentence1']
            label = example['label'] #.strip().split()
            answer = 1 if label == "Tuberculosis" else 0         
            official_train.append(dict(question=question, answer=answer))

        for example in tqdm.tqdm(hf_official_test):
            question = example['sentence1']
            label = example['label'] #.strip().split()
            answer = 1 if label == "Tuberculosis" else 0
            official_test.append(dict(question=question, answer=answer))

        rng = random.Random(0)
        rng.shuffle(official_train)
        rng = random.Random(0)
        rng.shuffle(official_test)
        trainset = official_train #[:200]
        devset = official_train #[200:500]
        testset = official_test[:]
        if split == "train":
            self.data = trainset
        elif split == "val":
            self.data = devset
        elif split == "test":
            self.data = testset



class DIAGNO_DSPy2(DIAGNO):
    def __init__(self, root:str=None, split: str="train"):
        """DSPy splits for the diagnosis dataset with balanced samples."""
        import tqdm
        import random
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        dataset = load_dataset("ninaa510/diagnosis-text", 'default', cache_dir=root)
        hf_official_train = dataset['train']
        hf_official_test = dataset['test']
        
        def process_and_balance_data(dataset, n_samples=100):
            positive_examples = []
            negative_examples = []
            
            for example in tqdm.tqdm(dataset):
                question = example['sentence1']
                label = example['label']
                answer = 1 if label == "Tuberculosis" else 0
                
                if answer == 1:
                    positive_examples.append(dict(question=question, answer=answer))
                else:
                    negative_examples.append(dict(question=question, answer=answer))
            
            # Randomly sample n instances from each class
            rng = random.Random(0)  # For reproducibility
            balanced_data = (
                rng.sample(positive_examples, n_samples) +
                rng.sample(negative_examples, n_samples)
            )
            rng.shuffle(balanced_data)  # Shuffle the combined data
            return balanced_data
        
        # Process train and test sets
        all_train_data = process_and_balance_data(hf_official_train, n_samples=100)
        all_test_data = process_and_balance_data(hf_official_test, n_samples=100)
        
        # Split train into train and validation
        trainset = all_train_data
        devset = all_train_data
        testset = all_test_data
        
        if split == "train":
            self.data = trainset
        elif split == "val":
            self.data = devset
        elif split == "test":
            self.data = testset
            
class DIAGNO_DSPy3(DIAGNO):
    def __init__(self, root: str = None, split: str = "train"):
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
            rng = random.Random(42)
            return rng.sample(pos, n_per_class) + rng.sample(neg, n_per_class)

        # Create balanced datasets
        full_balanced_train = balance_data(train_data, 300)  # 150 per class -> 300
        balanced_val = balance_data(train_data, 50)          # 50 per class  -> 100
        balanced_test = balance_data(test_data, 300)         # 100 per class -> 200

        # Shuffle for randomness
        rng = random.Random(42)
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