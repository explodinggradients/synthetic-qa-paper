import chunk
from pydantic import BaseModel, field_validator
from typing import Dict, Any, Optional

import os
import importlib
import inspect


class ComponentConfig(BaseModel):
    class_path: str
    params: Dict[str, Any] = {}

    @field_validator("class_path")
    @classmethod
    def validate_class_path(cls, v):
        """
        Validates that the class_path can be imported and the class exists.
        """
        try:
            module_path, class_name = v.rsplit(".", 1)
            module = importlib.import_module(module_path)
            getattr(module, class_name)
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"Cannot import '{v}': {e}")
        return v

    @field_validator("params")
    @classmethod
    def validate_params(cls, v, info):
        """
        Validates that the params match the class's __init__ signature.
        """
        class_path = info.data.get("class_path")
        if class_path:
            try:
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                target_class = getattr(module, class_name)
                sig = inspect.signature(target_class.__init__)
                valid_params = sig.parameters.keys()
                if 'kwargs' in valid_params:
                    return v
                for param in v:
                    if param not in valid_params:
                        raise ValueError(
                            f"Invalid parameter '{param}' for class '{class_path}'"
                        )
            except Exception as e:
                raise ValueError(f"Error validating params for '{class_path}': {e}")
        return v


class Config(BaseModel):
    directory: str
    text_splitter: ComponentConfig = ComponentConfig(
        class_path="langchain_text_splitters.RecursiveCharacterTextSplitter",
        params={"chunk_size": 512, "chunk_overlap": 100},
    )
    embedding_model: ComponentConfig = ComponentConfig(
        class_path="langchain_openai.OpenAIEmbeddings", params={}
    )
    llm: ComponentConfig = ComponentConfig(
        class_path="langchain_openai.ChatOpenAI", params={"model": "gpt-4o"}
    )
    retriever: Optional[ComponentConfig] = None
    
    @field_validator("directory")
    @classmethod
    def validate_directory(cls, v):
        """
        Validates that the directory exists.
        """
        if not os.path.exists(v):
            raise ValueError(f"Directory '{v}' does not exist.")
        return v


def instantiate_component(component_config: ComponentConfig):
    module_path, class_name = component_config.class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**component_config.params)
