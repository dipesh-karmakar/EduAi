import importlib


def load(config):
    parts = {}
    for name, cfg in config['components'].items():
        if cfg.get('enabled', False):
            module = importlib.import_module('components.' + name)
            parts[name] = module.build(config, parts)
    return parts
