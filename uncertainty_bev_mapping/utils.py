import os

def aggregate_dicts(dicts, fn):
    # All the dicts must have the same field.
    aggregated = {}
    for k, v in dicts[0]:
        aggregated[k] = []
    
    for d in dicts:
        for k, v in d:
            aggregated[k].append(v)
    for k, v in aggregated:
        if isinstance(v[0], dict):
            aggregated[k] = aggregate_dicts(v[0], fn)
        elif isinstance(v[0], list):
            aggregated[k] = aggregate_lists(v[0], fn)
        else:
            aggregated[k] = fn(v)
    return aggregated

def aggregate_lists(lists, fn):
    aggregated = []
    for v in lists[0]:
        aggregated.append([])
    for li, l in enumerate(lists):
        for vi, v in enumerate(l):
            aggregated[vi].append(v)
    for vi, v in enumerate(aggregated):
        if isinstance(v[0], dict):
            aggregated[vi] = aggregate_dicts(v, fn)
        elif isinstance(v[0], list):
            aggregated[vi] = aggregate_lists(v, fn)
        else:
            aggregated[vi] = fn(v)
    return aggregated

def split_path_into_folders(path):
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder:
            folders.append(folder)
        else:
            if path:
                folders.append(path)
            break
    folders.reverse()
    return folders
