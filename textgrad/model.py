from typing import Union
from textgrad.variable import Variable
from textgrad.autograd import LLMCall
from textgrad.autograd.function import Module
from textgrad.engine import EngineLM, get_engine
from .config import SingletonBackwardEngine

class BlackboxLLM(Module):
    def __init__(self, engine: Union[EngineLM, str] = None, system_prompt: Union[Variable, str] = None):
        """
        Initialize the LLM module.

        :param engine: The language model engine to use.
        :type engine: EngineLM
        :param system_prompt: The system prompt variable, defaults to None.
        :type system_prompt: Variable, optional
        """
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        if isinstance(system_prompt, str):
            system_prompt = Variable(system_prompt, requires_grad=False, role_description="system prompt for the language model")
        self.system_prompt = system_prompt
        self.llm_call = LLMCall(self.engine, self.system_prompt)

    def parameters(self):
        """
        Get the parameters of the blackbox LLM.

        :return: A list of parameters.
        :rtype: list
        """
        params = []
        if self.system_prompt:
            params.append(self.system_prompt)
        return params

    def forward(self, x: Variable) -> Variable:
        """
        Perform an LLM call.

        :param x: The input variable.
        :type x: Variable
        :return: The output variable.
        :rtype: Variable
        """     
        return self.llm_call(x)


import re

import re
from tqdm import tqdm

class DeepSeekLLM(Module):
    def __init__(self, engine: Union[EngineLM, str] = None, system_prompt: Union[Variable, str] = None):
        """
        Initialize the DeepSeek LLM module.

        :param engine: The language model engine to use.
        :type engine: EngineLM
        :param system_prompt: The system prompt variable, defaults to None.
        :type system_prompt: Variable, optional
        :param num_predict: Maximum number of tokens to predict in the response, defaults to 50.
        :type num_predict: int, optional
        """
        if engine is None:
            engine = SingletonBackwardEngine().get_engine()
            if engine is None:
                raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        
        if isinstance(engine, str):
            engine = get_engine(engine)
        
        self.engine = engine
        
        if isinstance(system_prompt, str):
            system_prompt = Variable(system_prompt, requires_grad=False, role_description="system prompt for the language model")
        
        self.system_prompt = system_prompt
        #self.num_predict = num_predict
        self.llm_call = LLMCall(self.engine, self.system_prompt) #, num_predict=self.num_predict)

    def parameters(self):
        """
        Get the parameters of the DeepSeek LLM.

        :return: A list of parameters.
        :rtype: list
        """
        params = []
        if self.system_prompt:
            params.append(self.system_prompt)
        return params

    def forward(self, x: Variable) -> Variable:
        """
        Perform an LLM call and remove unnecessary <think>...</think> parts.

        :param x: The input variable.
        :type x: Variable
        :return: The output variable with <think>...</think> removed.
        :rtype: Variable
        """     
        response = self.llm_call(x)
        if isinstance(response, Variable) and isinstance(response.value, str):
            response.value = re.sub(r"<think>.*?</think>", "", response.value, flags=re.DOTALL).strip()
        return response
