"""
Model Factory for Dynamic Model Creation
Supports multiple model frameworks through configuration
"""

import importlib
import sys
from typing import Any, Dict, Type
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import get_model_config, ModelConfig, DEFAULT_MODEL_KEY


class ModelFactory:
    """Factory class for creating models dynamically"""
    
    _model_cache: Dict[str, Type] = {}
    
    @classmethod
    def get_model_class(cls, model_key: str) -> Type:
        """Get model class by key"""
        if model_key in cls._model_cache:
            return cls._model_cache[model_key]
        
        config = get_model_config(model_key)
        
        # Import model module dynamically
        module_path = config.model_file.replace('/', '.').replace('.py', '')
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Could not import model module {module_path}: {e}")
        
        # Get model class
        model_class_name = f"{config.model_class}ForCausalLM"
        
        if not hasattr(module, model_class_name):
            raise AttributeError(f"Model class {model_class_name} not found in {module_path}")
        
        model_class = getattr(module, model_class_name)
        cls._model_cache[model_key] = model_class
        
        return model_class
    
    @classmethod
    def create_model(cls, model_key: str = None, **kwargs) -> Any:
        """Create model instance"""
        if model_key is None:
            model_key = DEFAULT_MODEL_KEY
            
        config = get_model_config(model_key)
        model_class = cls.get_model_class(model_key)
        
        # Merge default parameters with provided kwargs
        # User-provided kwargs should override default params
        model_kwargs = kwargs.copy()
        
        # Only add default params that are not already specified by user
        for key, value in config.default_params.items():
            if key not in model_kwargs:
                model_kwargs[key] = value
        
        return model_class, model_kwargs
    
    @classmethod
    def get_model_info(cls, model_key: str = None) -> Dict[str, Any]:
        """Get model information"""
        if model_key is None:
            model_key = DEFAULT_MODEL_KEY
            
        config = get_model_config(model_key)
        return {
            "model_name": config.model_name,
            "model_key": config.model_key,
            "description": config.description,
            "default_params": config.default_params,
            "model_file": config.model_file
        }
    
    @classmethod
    def clear_cache(cls):
        """Clear model cache"""
        cls._model_cache.clear()


def create_model_from_config(model_key: str = None, **kwargs):
    """Convenience function to create model"""
    return ModelFactory.create_model(model_key, **kwargs)


def get_available_models():
    """Get list of available models"""
    from configs.model_config import list_available_models
    return list_available_models() 