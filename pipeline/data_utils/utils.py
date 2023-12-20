import random

def idx2option(idx: int, style="upper", deco="dot"):
    """
    idx: [0, N-1]
    style: upper, lower, num
    deco: None, paren, dot, rparen
    """
    idx = {
        "upper": chr(ord("A") + idx),
        "lower": chr(ord("a") + idx),
        "num": f"{idx + 1}",
    }[style]

    idx = {
        None: "{idx}",
        "paren": "({idx})",
        "dot": "{idx}.",
        "rparen": "{idx})",
    }[deco].format(idx=idx)

    return idx


def optionize(
    options: list[str],
    answer_idx: int,
    shuffle=False,
    aug_idx_style=False,
    include_answer_str=False,
    sep="\n"
) -> (str, str):
    """Convert options (list of str) to option string.
    This process also includes:
    - option shuffling
    - index augmentation
    Args:
        options (list[str])
        answer_idx (int)
        shuffle (bool): shuffle options
        aug_idx_style (bool): randomly choose index style
            Aug examples: (1) / 1. / (A) / A.
        include_answer_str (bool): include answer string
            False: A
            True: A. {answer}
    Return:
        (option_str, answer_str)
    """
    if isinstance(options, str):
        # already optionized
        return options

    answer = options[answer_idx]
    if shuffle:
        random.shuffle(options)
        answer_idx = options.index(answer)

    if not aug_idx_style:
        style = "upper"
        deco = "dot"
    else:
        style = random.choice(["upper", "lower", "num"])
        deco = random.choice(["paren", "dot", "rparen"])

    indices = [idx2option(i, style=style, deco=deco) for i in range(len(options))]
    answer_str = idx2option(answer_idx, style=style, deco=None)
    if include_answer_str:
        answer_str = f"{answer_str}. {answer}"

    options_with_index = [
        f"{idx} {option}"
        for idx, option in zip(indices, options)
    ]
    option_str = sep.join(options_with_index)
    return option_str, answer_str
