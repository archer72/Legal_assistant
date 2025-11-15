"""Hugging Face pipeline -> LangChain wrapper LLM"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
from typing import List
from utils.logger import get_logger

logger = get_logger('generator')


class Generator:
    def __init__(self, model_path: str = None, max_new_tokens: int = 512, temperature: float = 0.0):
        model_path = model_path or os.getenv('HF_LLM_PATH')
        if not model_path:
            raise ValueError('HF model path must be provided via model_path or HF_LLM_PATH env var')

        device = 0 if torch.cuda.is_available() else -1
        logger.info(f'Loading HF model {model_path} on device {device}')

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')

        # text-generation pipeline
        self.pipe = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
        )

        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def generate(self, prompt: str):
         # returns LangChain-style string output
        return self.llm(prompt)
