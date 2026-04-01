import yaml
import os

#a simple function to load configs from a yaml file
def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    #make paths absolute relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    if 'paths' in config:
        for key, val in config['paths'].items():
            if not os.path.isabs(val):
                config['paths'][key] = os.path.join(project_root, val)
                
    return config
    