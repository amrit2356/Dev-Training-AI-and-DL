import ast
import json
from types import SimpleNamespace
import torch
import multiprocessing

def parse_input(args):
    with open(args.config_path, 'r') as data:
        config = json.load(data, object_hook=lambda d: SimpleNamespace(**d))

    if ast.literal_eval(config.normalization_param.num_workers) is None:
        config.normalization_param.num_workers = multiprocessing.cpu_count()

    if ast.literal_eval(config.normalization_param.batch_size) is None:
        config.normalization_param.batch_size = config.normalization_param.num_workers

    if ast.literal_eval(config.normalization_param.device) is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config.normalization_param.device = torch.device(device)
    return config
