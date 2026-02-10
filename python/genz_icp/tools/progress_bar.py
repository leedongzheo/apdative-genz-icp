from tqdm.auto import trange


def get_progress_bar(first_scan, last_scan):
    return trange(first_scan, last_scan, unit=" frames", dynamic_ncols=True)
