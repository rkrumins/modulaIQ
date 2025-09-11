"""Configuration management for the multi-agent framework."""

import os
import yaml
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    api_key_env: str
    base_url: Optional[str] = None
    default_model: str
    available_models: list[str]


class MemoryConfig(BaseModel):
    """Configuration for memory backends."""
    redis: Optional[Dict[str, Any]] = None
    sqlite: Optional[Dict[str, Any]] = None


class SupervisorConfig(BaseModel):
    """Configuration for the supervisor."""
    name: str = "Supervisor"
    host: str = "localhost"
    port: int = 8000
    llm_provider: str = "groq"
    model: str = "llama-3.1-8b-instant"
    max_iterations: int = 10
    memory_backend: str = "redis"


class AgentRegistryConfig(BaseModel):
    """Configuration for agent registry."""
    discovery_port: int = 8001
    health_check_interval: int = 30
    max_retries: int = 3


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/agent_orchestrator.log"


class Config(BaseModel):
    """Main configuration class."""
    supervisor: SupervisorConfig
    llm_providers: Dict[str, LLMProviderConfig]
    memory: MemoryConfig
    agent_registry: AgentRegistryConfig
    logging: LoggingConfig

    @classmethod
    def load_from_yaml(cls, config_path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def get_llm_provider_config(self, provider_name: str) -> LLMProviderConfig:
        """Get configuration for a specific LLM provider."""
        if provider_name not in self.llm_providers:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
        return self.llm_providers[provider_name]
    
    def get_api_key(self, provider_name: str) -> str:
        """Get API key for a specific provider from environment variables."""
        provider_config = self.get_llm_provider_config(provider_name)
        api_key = os.getenv(provider_config.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found for provider {provider_name}. "
                           f"Please set {provider_config.api_key_env} environment variable.")
        return api_key
