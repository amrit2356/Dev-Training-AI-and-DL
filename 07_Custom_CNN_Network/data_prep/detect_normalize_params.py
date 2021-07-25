import ast
from math import ceil

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
from torchvision import transforms

from .team_dataset import TeamDataset


class FiniteRandomSampler(data.Sampler):
    def __init__(self, data_source, num_samples):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).tolist()[: self.num_samples])

    def __len__(self):
        return self.num_samples


class RunningAverage:
    def __init__(self, num_channels=3, **meta):
        self.num_channels = num_channels
        self.avg = torch.zeros(num_channels, **meta)

        self.num_samples = 0

    def update(self, vals):
        batch_size, num_channels = vals.size()

        if num_channels != self.num_channels:
            raise RuntimeError

        updated_num_samples = self.num_samples + batch_size
        correction_factor = self.num_samples / updated_num_samples

        updated_avg = self.avg * correction_factor
        updated_avg += torch.sum(vals, dim=0) / updated_num_samples

        self.avg = updated_avg
        self.num_samples = updated_num_samples

    def tolist(self):
        return self.avg.detach().cpu().tolist()

    def __str__(self):
        return "[" + ", ".join([f"{val:.3f}" for val in self.tolist()]) + "]"

    def __make_reproducible(self, seed):
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def param_calculation(self, config, dataset_path, csv_path):
        if ast.literal_eval(config.normalization_param.seed) is not None:
            self.__make_reproducible(config.normalization_param.seed)

        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((160, 64)), transforms.ToTensor()]
        )

        labels = pd.read_csv(csv_path)
        train_data, _ = train_test_split(labels, stratify=labels.cls, test_size=0.2)

        dataset = TeamDataset(train_data, dataset_path, transform=transform)

        num_samples = ast.literal_eval(config.normalization_param.num_samples)
        if num_samples is None:
            num_samples = len(dataset)
        if num_samples < len(dataset):
            sampler = FiniteRandomSampler(dataset, num_samples)
        else:
            sampler = data.SequentialSampler(dataset)

        loader = data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=config.normalization_param.num_workers,
            batch_size=config.normalization_param.batch_size,
        )

        running_mean = RunningAverage(device=config.normalization_param.device)
        running_std = RunningAverage(device=config.normalization_param.device)
        num_batches = ceil(num_samples / config.normalization_param.batch_size)

        with torch.no_grad():
            for batch, (images, _) in enumerate(loader, 1):
                images = images.to(config.normalization_param.device)
                images_flat = torch.flatten(images, 2)

                mean = torch.mean(images_flat, dim=2)
                running_mean.update(mean)

                std = torch.std(images_flat, dim=2)
                running_std.update(std)

                if not config.normalization_param.quiet and batch % config.normalization_param.print_freq == 0:
                    print(
                        (
                            f"[{batch:6d}/{num_batches}] "
                            f"mean={running_mean}, std={running_std}"
                        )
                    )

        print(f"mean={running_mean}, std={running_std}")

        return running_mean.tolist(), running_std.tolist()
