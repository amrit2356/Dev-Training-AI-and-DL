import os
import cv2

from torch.utils.data import Dataset

class TeamDataset(Dataset):
    def __init__(self, data, path, transform=None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, 'train', img_name)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data)
