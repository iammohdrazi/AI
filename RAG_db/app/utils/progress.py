from tqdm import tqdm


def progress_bar(iterable, desc="", unit="item", leave=False):
    return tqdm(
        iterable,
        desc=desc,
        unit=unit,
        leave=leave,
        ncols=100
    )
