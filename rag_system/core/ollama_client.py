"""Ollama client for LLM interactions."""

import logging
from typing import Dict, Any, Optional
import ollama
from .config import config

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama LLM."""
    
    def __init__(self, base_url: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL
            model_name: Name of the model to use
        """
        self.base_url = base_url or config.ollama_base_url
        self.model_name = model_name or config.ollama_model
        self.client = ollama.Client(host=self.base_url)
        
    def check_model_availability(self) -> bool:
        """Check if the specified model is available."""
        try:
            models = self.client.list()
            available_models = [model['name'] for model in models['models']]
            return self.model_name in available_models
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    def pull_model(self) -> bool:
        """Pull the model if not available."""
        try:
            logger.info(f"Pulling model: {self.model_name}")
            self.client.pull(self.model_name)
            return True
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate response using Ollama.
        
        Args:
            prompt: The user query
            context: Retrieved context for RAG
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            # Construct the full prompt
            if context:
                full_prompt = f"""Based on the following context, please answer the question:

Context:
{context}

Question: {prompt}

Answer:"""
            else:
                full_prompt = prompt
            
            # Generate response
            response = self.client.generate(
                model=self.model_name,
                prompt=full_prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens or config.max_tokens,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Unable to generate response - {str(e)}"
    
    def chat(self, messages: list, temperature: float = 0.7) -> str:
        """Chat using conversation format.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': temperature,
                    'num_predict': config.max_tokens,
                }
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error: Unable to chat - {str(e)}"
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of Ollama service.
        
        Returns:
            Health status information
        """
        try:
            models = self.client.list()
            model_available = self.check_model_availability()
            
            return {
                'status': 'healthy',
                'base_url': self.base_url,
                'model_name': self.model_name,
                'model_available': model_available,
                'available_models': [model['name'] for model in models['models']]
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'base_url': self.base_url,
                'model_name': self.model_name
            }