"""
Model Configuration Registry
Supports multiple model frameworks with configurable parameters
"""

from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Model configuration class"""
    model_name: str  # Model display name
    model_key: str   # Model key for internal use
    model_class: str # Model class name
    model_file: str  # Model file path
    description: str # Model description
    default_params: Dict[str, Any] = field(default_factory=dict)  # Default parameters
    
# Removed unused get_class_name method

# Model registry
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "PixelLM": ModelConfig(
        model_name="PixelLM",
        model_key="PixelLM", 
        model_class="PixelLM",
        model_file="model/PixelLM.py",
        description="Original PixelLM model for pixel-level reasoning",
        default_params={}  # No default params to avoid conflicts
    ),
    # Add more models here as needed
    "GSVA": ModelConfig(
        model_name="GSVA",
        model_key="GSVA", 
        model_class="GSVA",
        model_file="model/GSVA.py",
        description="Original GSVA model for generalized segmentation",
        default_params={}  # No default params to avoid conflicts
    ),
    "SESAME": ModelConfig(
        model_name="SESAME",
        model_key="SESAME",
        model_class="SESAME",
        model_file="model/SESAME.py",
        description="Original SESAME model for semantic segmentation",
        default_params={}  # No default params to avoid conflicts
    ),
    "READ": ModelConfig(
        model_name="READ",
        model_key="READ",
        model_class="READ",
        model_file="model/READ.py",
        description="Original READ model for semantic segmentation",
        default_params={}  # No default params to avoid conflicts
    ),
    "LISA": ModelConfig(
        model_name="LISA",
        model_key="LISA",
        model_class="LISA",
        model_file="model/LISA.py",
        description="Original LISA model for semantic segmentation",
        default_params={}  # No default params to avoid conflicts
    ),
    "UGround": ModelConfig(
        model_name="UGround",
        model_key="UGround",
        model_class="UGround",
        model_file="model/UGround.py",
        description="Original UGround model for semantic segmentation",
        default_params={}  # No default params to avoid conflicts
    ),
}

def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key"""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_key]

def list_available_models() -> Dict[str, str]:
    """List all available models"""
    return {key: config.model_name for key, config in MODEL_REGISTRY.items()}

def register_model(model_key: str, config: ModelConfig):
    """Register a new model configuration"""
    MODEL_REGISTRY[model_key] = config

# Default model key
DEFAULT_MODEL_KEY = "PixelLM"