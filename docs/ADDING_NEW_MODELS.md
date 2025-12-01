# Adding New Models to the Multi-Model Framework

This guide explains how to add new models to the PixelLM multi-model framework.

## Overview

The framework is designed to support multiple model architectures through a common interface. Each model can have its own specific parameters, initialization logic, and components while sharing common functionality.

## Step-by-Step Guide

### 1. Create Model Configuration

First, create a configuration for your new model in `configs/model_config.py`:

```python
from .model_config import ModelConfig

YOUR_MODEL_CONFIG = ModelConfig(
    model_name="YourModelName",
    model_key="your_model_key",  # Used in --model_key parameter
    model_class="YourModel",     # Will become YourModelForCausalLM
    model_file="model/YourModel.py",  # Path to your model implementation
    description="Description of your model",
    default_params={
        # Common parameters
        "vision_tower": "openai/clip-vit-large-patch14",
        "use_mm_start_end": True,
        "vision_tower_for_mask": True,
        "resize_vision_tower": True,
        "resize_vision_tower_size": 448,
        "train_mask_decoder": True,
        "out_dim": 256,
        
        # Your model-specific parameters
        "your_custom_param": "default_value",
        "another_param": 123,
    }
)
```

Then register it in the `MODEL_REGISTRY`:

```python
MODEL_REGISTRY["your_model_key"] = YOUR_MODEL_CONFIG
```

### 2. Implement Your Model

Create your model file (e.g., `model/YourModel.py`) using the template in `model/example_new_model_template.py`.

#### Key Components:

1. **Meta Model Class**: Handles core logic and module initialization
2. **Model Class**: Combines meta model with LLaVA's LlamaModel
3. **ForCausalLM Class**: Main class instantiated by ModelFactory

#### Required Methods:

- `initialize_vision_modules(config)`: Initialize vision-related components
- `initialize_model_specific_modules(config)`: Initialize model-specific components

### 3. Example Implementation Structure

```python
class YourModelMetaModel(BaseModelInterface, BaseModelMixin):
    def __init__(self, config, **kwargs):
        # Initialize your model
        pass
    
    def initialize_vision_modules(self, config):
        # Initialize CLIP, vision tower, etc.
        pass
    
    def initialize_model_specific_modules(self, config):
        # Initialize SAM, mask decoder, custom components
        pass

class YourModelModel(YourModelMetaModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        # Set up LLaVA configuration
        pass

class YourModelForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        # Main model class
        # Set default parameters
        # Initialize components
        pass
```

### 4. Usage

Once implemented, you can use your model with:

```bash
# Training
python train_ds.py --model_key your_model_key [other parameters]

# Chat interface
python chat.py --model_key your_model_key [other parameters]

# Web app
python app.py --model_key your_model_key [other parameters]
```

## Framework Features

### Automatic Parameter Merging
- Default parameters from config are merged with user-provided parameters
- User parameters take precedence over defaults
- No need to manually handle parameter conflicts

### Backward Compatibility
- Framework maintains compatibility with existing PixelLM code
- Legacy method names are preserved through compatibility methods
- Existing scripts work without modification

### Dynamic Loading
- Models are loaded dynamically based on configuration
- No need to modify core framework code when adding new models
- Supports hot-swapping between different model architectures

## Best Practices

1. **Inherit from Base Classes**: Use `BaseModelInterface` and `BaseModelMixin` for consistency
2. **Follow Naming Conventions**: Use consistent naming for classes and methods
3. **Document Parameters**: Clearly document all model-specific parameters
4. **Test Thoroughly**: Test your model with training, inference, and web interface
5. **Maintain Compatibility**: Ensure your model works with existing data loaders and evaluation scripts

## Common Patterns

### Custom Vision Processing
```python
def initialize_vision_modules(self, config):
    # Custom vision tower setup
    self.custom_vision_processor = CustomVisionProcessor(config)
    # Standard LLaVA vision setup is handled by parent class
```

### Custom Decoders
```python
def initialize_model_specific_modules(self, config):
    # Custom mask decoder
    self.custom_decoder = CustomMaskDecoder(config)
    # Standard text projection layers
    self.text_hidden_fcs = nn.ModuleList([...])
```

### Model-Specific Parameters
```python
default_params = {
    # Standard parameters
    "vision_tower": "openai/clip-vit-large-patch14",
    # Your custom parameters
    "custom_attention_heads": 8,
    "custom_fusion_method": "cross_attention",
}
```

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure all required modules are properly imported
2. **Parameter Conflicts**: Check that parameter names don't conflict with existing ones
3. **Initialization Order**: Make sure modules are initialized in the correct order
4. **Missing Methods**: Implement all required abstract methods from base classes

### Debugging Tips:

1. Use the model info function: `ModelFactory.get_model_info("your_model_key")`
2. Check parameter merging: Print kwargs in your model's `__init__` method
3. Verify module initialization: Add logging to initialization methods
4. Test incrementally: Start with a minimal implementation and add features gradually

## Examples

See `configs/example_other_model.py` and `model/example_new_model_template.py` for complete examples of how to implement new models in the framework. 