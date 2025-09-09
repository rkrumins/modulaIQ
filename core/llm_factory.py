"""Factory for creating LLM instances based on configuration."""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseLanguageModel

from .config import Config, LLMProviderConfig


class LLMFactory:
    """Factory class for creating LLM instances."""
    
    @staticmethod
    def create_llm(config: Config, provider_name: str, model: Optional[str] = None) -> BaseLanguageModel:
        """Create an LLM instance based on provider configuration."""
        provider_config = config.get_llm_provider_config(provider_name)
        api_key = config.get_api_key(provider_name)
        
        if model is None:
            model = provider_config.default_model
        
        if provider_name == "groq":
            return ChatGroq(
                model=model,
                groq_api_key=api_key,
                temperature=0.1
            )
        elif provider_name == "openai":
            return ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=0.1,
                base_url=provider_config.base_url
            )
        elif provider_name == "google":
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.1
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
    
    @staticmethod
    def create_supervisor_llm(config: Config) -> BaseLanguageModel:
        """Create LLM instance for supervisor."""
        return LLMFactory.create_llm(
            config, 
            config.supervisor.llm_provider, 
            config.supervisor.model
        )
