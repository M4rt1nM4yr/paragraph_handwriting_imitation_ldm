########################################################################################################################
# modified code FROM https://github.com/CompVis/latent-diffusion
# Paper: https://arxiv.org/pdf/2112.10752.pdf
########################################################################################################################
import importlib
from omegaconf import OmegaConf
import os


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params

#TODO is this name problematic? It would break a lot...
def get_yaml(application,filename,project_name="mt_handwriting-diffusion",config_directory="configs"):
    mypath = os.getcwd()
    path = mypath[:mypath.find(project_name) + len(project_name)]
    return os.path.join(path, config_directory,application,filename)

def instantiate_completely(application,filename,**kwargs):
    file = get_yaml(application,filename)
    cfg = OmegaConf.load(file)
    return instantiate_from_config(cfg,**kwargs)

def instantiate_from_config(config,**kwargs):
    if "ckpt" in config:
        return get_obj_from_str(config["target"]).load_from_checkpoint(checkpoint_path=config["ckpt"],
                                                                       **(config.get("params", dict())),
                                                                       strict=False,**kwargs )

    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()),**kwargs )


def instantiate_from_config_function(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])


def instantiate_from_ckpt(config):
    return None

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
