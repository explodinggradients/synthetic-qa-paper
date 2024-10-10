from pydantic import BaseModel, field_validator
from typing import Dict, Any, Optional, Literal, List
import json

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
                if "kwargs" in valid_params:
                    return v
                for param in v:
                    if param not in valid_params:
                        raise ValueError(
                            f"Invalid parameter '{param}' for class '{class_path}'"
                        )
            except Exception as e:
                raise ValueError(f"Error validating params for '{class_path}': {e}")
        return v


class RetrieverConfig(BaseModel):
    type: Literal["vanilla", "multi_query"] = "vanilla"
    params: Dict[str, Any] = {"search_kwargs": {"k": 2}}


class DirectoryConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {"sample_size": 1, "show_progress": True}


class Config(BaseModel):
    directory: DirectoryConfig
    experiment_name: str = "default_experiment"
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
    retriever: RetrieverConfig = RetrieverConfig()
    model_prompt: str = "Answer the user query based on the context"

    @field_validator("directory")
    @classmethod
    def validate_directory(cls, v):
        """
        Validates that the directory exists.
        """
        if not os.path.exists(v.name):
            raise ValueError(f"Directory '{v.name}' does not exist.")
        return v

    @classmethod
    def load(cls, config_name_or_path: str):
        config = json.load(open(config_name_or_path))
        return cls(**config)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)


def instantiate_component(component_config: ComponentConfig):
    module_path, class_name = component_config.class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**component_config.params)
