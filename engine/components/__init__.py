# Description: Loads enabled engine components from configuration.
# Author: Dipesh Karmakar <dipesh.karmakar@gmail.com>
# Modifications:
# - f9dc837: Added dynamic component loading.

import importlib


def load(config):
    parts = {}
    for name, cfg in config['components'].items():
        if cfg.get('enabled', False):
            module = importlib.import_module('components.' + name)
            parts[name] = module.build(config, parts)
    return parts
