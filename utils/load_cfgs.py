import yaml

from argparse import Namespace
from typing import Any, Dict

def dict_to_namespace(data: Dict[str, Any]) -> Namespace:
    namespace = Namespace()
    for key, value in data.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        elif isinstance(value, list):
            setattr(namespace, key, [
                dict_to_namespace(item) if isinstance(item, dict) else item
                for item in value
            ])
        else:
            setattr(namespace, key, value)
    return namespace

def load_yaml_as_argparse(yaml_path: str) -> Namespace:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    return dict_to_namespace(yaml_data)