import yaml
import os

#a simple function to load configs from a yaml file
def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)