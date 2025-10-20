import hashlib
import diskcache as dc
from abc import ABC, abstractmethod

class EngineLM(ABC):
    system_prompt: str = "You are a helpful, creative, and smart assistant."
    model_string: str
    @abstractmethod
    def generate(self, prompt, system_prompt=None, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


class CachedEngine2:
    def __init__(self, cache_path):
        super().__init__()
        self.cache_path = cache_path
        self.cache = {}  # In-memory only, never persisted

    def _hash_prompt(self, prompt: str):
        import hashlib
        return hashlib.sha256(f"{prompt}".encode()).hexdigest()

    def _check_cache(self, prompt: str):
        # Always returns None, disables cache
        return None

    def _save_cache(self, prompt: str, response: str):
        # Do nothing, disables cache
        pass

    def __getstate__(self):
        # Remove the cache from the state before pickling
        state = self.__dict__.copy()
        if 'cache' in state:
            del state['cache']
        return state

    def __setstate__(self, state):
        # Restore the cache after unpickling as empty dict
        self.__dict__.update(state)
        self.cache = {}


class CachedEngine:
    def __init__(self, cache_path):
        super().__init__()
        self.cache_path = cache_path
        self.cache = dc.Cache(cache_path)
        #self.cache = {}
    def _hash_prompt(self, prompt: str):
        return hashlib.sha256(f"{prompt}".encode()).hexdigest()

    def _check_cache(self, prompt: str):
        if prompt in self.cache:
            return self.cache[prompt]
        else:
            return None

    def _save_cache(self, prompt: str, response: str):
        self.cache[prompt] = response

    def __getstate__(self):
        # Remove the cache from the state before pickling
        state = self.__dict__.copy()
        del state['cache']
        return state

    def __setstate__(self, state):
        # Restore the cache after unpickling
        self.__dict__.update(state)
        self.cache = dc.Cache(self.cache_path)
