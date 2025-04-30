import importlib


def load_instance(config):
    if isinstance(config, list):
        return [get_obj_from_str(cfg["target"])(**cfg.get("params", dict())) for cfg in config]
    else:
        if not "target" in config:
            raise KeyError("Expected key `target` to instantiate.")
        return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
