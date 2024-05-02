# NOTE on template patterns
# Fixed texts: role tokens (Human, AI), image tokens, and system messages.
# You can control (instruction, input, target) using PATTERN_DICT.
#
# ```
# The following is a conversation between a curious human and AI assistant. {instruction}
# Human: <image>
# Human: {input}
# AI: {target}
# ```


def parse_pattern(pattern_dict: dict) -> dict:
    """Parsing patterns with pre-defined rules:
    - `-1` indicates "same as above"
    """
    new_pattern_dict = {}
    for k, patterns in pattern_dict.items():
        if isinstance(patterns, str):
            new_pattern_dict[k] = pattern_dict[patterns]
            continue

        new_patterns = []
        for i, (inst, inputs, targets) in enumerate(patterns):
            if inst == -1:
                assert i > 0
                inst = patterns[i - 1][0]

            new_patterns.append((inst, inputs, targets))

        new_pattern_dict[k] = new_patterns

    return new_pattern_dict


class Template:
    _registry = {}

    def __init__(self, pattern_dict: dict, pattern_map: dict):
        self.pattern_dict = parse_pattern(pattern_dict)
        self.pattern_map = pattern_map
        self.data2pattern = {}
        for pattern_name, dset_names in pattern_map.items():
            for dset_name in dset_names:
                assert dset_name not in self.data2pattern, "Duplicated dset name exists"
                self.data2pattern[dset_name] = pattern_name

    def get_pattern(self, dset_name):
        if dset_name in self.data2pattern:
            pattern_key = self.data2pattern[dset_name]
            return self.pattern_dict[pattern_key]
        elif dset_name in self.pattern_dict:
            return self.pattern_dict[dset_name]
        else:
            return None

    @classmethod
    def register(cls, name: str, pattern_dict: dict, pattern_map: dict):
        template = cls(pattern_dict, pattern_map)
        cls._registry[name] = template

    @classmethod
    def get(cls, name: str):
        return cls._registry[str(name)]
