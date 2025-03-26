import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class DeepGlobeDataset(Dataset):
    def __init__(self, metadata_file, root_dir, transform=None):
        self.metadata = pd.read_csv(metadata_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_dict = {
            'urban_land': (0, 255, 255),
            'agriculture_land': (255, 255, 0),
            'rangeland': (255, 0, 255),
            'forest_land': (0, 255, 0),
            'water': (0, 0, 255),
            'barren_land': (255, 255, 255),
            'unknown': (0, 0, 0)
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.metadata.iloc[idx, 2])
        image = Image.open(img_name).convert("RGB")
        
        if self.metadata.iloc[idx, 1] == 'train':
            mask_name = os.path.join(self.root_dir, self.metadata.iloc[idx, 3])
            mask = Image.open(mask_name)
        else:
            mask = Image.new('RGB', image.size, (0, 0, 0))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = self.rgb_to_class_index(mask)

        return image, mask

    def rgb_to_class_index(self, mask):
        mask_np = np.array(mask)
        _, height, width = mask_np.shape
        class_mask = np.zeros((height, width), dtype=np.int64)
        for class_idx, color in enumerate(self.class_dict.values()):
            color_np = np.array(color)[:, None, None]
            class_mask[(mask_np == color_np).all(axis=0)] = class_idx
        return torch.from_numpy(class_mask)
