import os

from omegaconf import DictConfig, OmegaConf

import utils


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)

    @classmethod
    def from_omegaconf(cls, data, resolve=True):
        """Load AttrDict from Omegaconf Container"""
        data = OmegaConf.to_container(data, resolve=resolve)
        return cls.from_nested_dicts(data)

    @classmethod
    def from_nested_dicts(cls, data):
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})

    def asdict(self):
        def _asdict(data):
            if not isinstance(data, dict):
                return data
            else:
                return {key: _asdict(data[key]) for key in data}

        return _asdict(self)

    def dumps(self) -> str:
        return OmegaConf.to_yaml(self.asdict())


def save_config(cfg, only_main_process=True):
    """Save config to `config.output_dir/exp_config.yaml`.
    """
    if only_main_process and not utils.is_main_process():
        return

    output_dir = cfg.output_dir
    if isinstance(cfg, AttrDict):
        cfg = cfg.asdict()  # AttrDict does not work with OmegaConf.to_yaml

    out_conf_dir = os.path.dirname(output_dir)
    os.makedirs(out_conf_dir, exist_ok=True)
    with open(os.path.join(output_dir, "exp_config.yaml"), "w") as fout:
        fout.write(OmegaConf.to_yaml(cfg) + "\n")


def set_config(cfg: DictConfig, save: bool = False) -> AttrDict:
    # convert DictConfig to AttrDict
    # - it is slow to access DictConfig
    # - DictConfig makes an unresolved error:
    #   `RuntimeError: DataLoader worker (pid 7103) is killed by signal: Aborted`.

    OmegaConf.resolve(cfg)  # in-place update

    if save:
        # config loaded by hydra is saved to <output_dir>/exp_config.yaml
        save_config(cfg)

    cfg = OmegaConf.to_container(cfg)
    cfg = AttrDict.from_nested_dicts(cfg)
    return cfg


def load_config(cfg_path: str, resolve=True) -> AttrDict:
    cfg = OmegaConf.load(cfg_path)
    cfg = AttrDict.from_omegaconf(cfg, resolve=resolve)
    return cfg
