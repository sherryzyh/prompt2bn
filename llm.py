"""
BNSynth LLM Client Module

This module provides a unified interface for interacting with various LLM providers
used in BNSynth for Bayesian Network synthesis. Supports both LLM-based generation
(data-free structure creation) and LLM-enhanced refinement (intelligent structure optimization).

Supported providers:
- OpenAI (GPT models, O-series reasoning models)
- DeepSeek
- Google Gemini

API Key Configuration:
The module checks for API keys in the following order:
1. Environment variables (e.g., OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY)
2. .env file in the project root (if environment variable is not set)

The .env file will NOT override existing environment variables, allowing flexible configuration
for different deployment scenarios.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
import logging

class LLMClient:
    def __init__(self, model: str, logger=None):
        # Load API keys from .env file if not already in environment variables
        # load_dotenv() does NOT override existing environment variables
        load_dotenv()
        self.model = model
        self.client = None
        self.provider = None
        self.logger = logger
        
    def chat(self, messages):
        raise NotImplementedError("Subclasses must implement .chat()")

    def chat_structured(self, messages, response_format=None):
        raise NotImplementedError("Subclasses must implement .chat_structured()")

    @staticmethod
    def from_model(model: str, logger=None):
        model_lower = model.lower()
        if "deepseek" in model_lower:
            return DeepseekModel(model, logger)
        elif model_lower.startswith("o"):
            return OAIReasonModel(model, logger)
        elif model_lower.startswith("gpt"):
            return OAIChatModel(model, logger)
        elif model_lower.startswith("gemini"):
            return GeminiModel(model, logger)
        else:
            raise ValueError(f"Unknown model: {model}")

class DeepseekModel(LLMClient):
    def __init__(self, model: str, logger = None):
        super().__init__(model, logger)
        # Check for API key (already loaded from .env by parent __init__)
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found. Please set it as an environment variable "
                "or add it to your .env file."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.provider = "deepseek"

    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

class OAIReasonModel(LLMClient):
    def __init__(self, model: str, logger=None):
        super().__init__(model, logger)
        # Check for API key (already loaded from .env by parent __init__)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it as an environment variable "
                "or add it to your .env file."
            )
        self.client = OpenAI(api_key=api_key)
        self.provider = "openai"

    def chat(self, messages):
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "medium"},
            input=messages
        )
        return response.output_text

    def chat_structured(self, messages, response_format=None):
        response = self.client.chat.beta.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_format
        )
        return response.choices[0].message.content

class OAIChatModel(LLMClient):
    def __init__(self, model: str, logger=None):
        super().__init__(model, logger)
        # Check for API key (already loaded from .env by parent __init__)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it as an environment variable "
                "or add it to your .env file."
            )
        self.client = OpenAI(api_key=api_key)
        self.provider = "openai"

    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content 
    
    def chat_structured(self, messages, response_format=None):
        self.logger.error("Structured Output is not supported for {self.model}")
        raise NotImplementedError("Structured Output is not supported for {self.model}")


class GeminiModel(LLMClient):
    def __init__(self, model: str, logger=None):
        super().__init__(model, logger)
        # Check for API key (already loaded from .env by parent __init__)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it as an environment variable "
                "or add it to your .env file."
            )
        
        genai.configure(api_key=api_key)
        self.provider = "google"
        # The client is just a reference to the `genai` module or can be kept as None
        # since we will create the model instance within the chat method.
        # This makes the class more flexible and stateless per chat call.
        self.client = None

    def _convert_openai_messages_to_gemini_contents(self, messages):
        """
        Converts OpenAI-style messages to Gemini-style contents and extracts
        the system instruction.
        Returns a tuple: (system_instruction, contents)
        """
        system_instruction = None
        contents = []

        if messages and messages[0]['role'] == 'system':
            system_instruction = messages[0]['content']
            conversation_messages = messages[1:]
        else:
            conversation_messages = messages
        
        for msg in conversation_messages:
            role = "user" if msg['role'] == 'user' else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg['content']}]
            })
            
        return system_instruction, contents

    def chat(self, messages, *args, **kwargs):
        """
        Sends a list of messages to the Gemini API and returns the response text.
        """
        system_instruction, contents = self._convert_openai_messages_to_gemini_contents(messages)
        
        # Instantiate the model with the system instruction for this specific chat call.
        # This is the crucial fix. The GenerativeModel instance is created for each call
        # if a system instruction is present.
        if system_instruction:
            model_instance = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_instruction
            )
        else:
            model_instance = genai.GenerativeModel(model_name=self.model)

        response = model_instance.generate_content(
            contents=contents,
            *args,
            **kwargs,
        )
        
        return response.text

    def chat_structured(self, messages, response_format=None):
        # ... (same as before, or implement structured output with function calling)
        raise NotImplementedError(
            f"Structured chat for Gemini requires function calling, which is not implemented for {self.model}"
        )