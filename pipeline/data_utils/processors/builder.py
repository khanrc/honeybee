from collections import UserDict

from pipeline.data_utils.registry import Registry, build_from_cfg

PROCESSORS = Registry('processors')


class DefaultDict(UserDict):
    def __getitem__(self, key):
        return self.data.get(key, self.data["default"])


def build_processors(processors_cfg, **cfg):
    processors = {}
    for task, processor in processors_cfg.items():
        proc_cfg = processor | cfg
        processors[task] = build_from_cfg(proc_cfg, PROCESSORS)

    processors = DefaultDict(processors)
    return processors
