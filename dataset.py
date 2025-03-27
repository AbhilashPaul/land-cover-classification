import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter

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

    def print_stats(self):
        print(f"\nDataset Statistics:")
        print(f"Number of images: {len(self)}")
        
        # Count class distribution
        all_labels = []
        for idx in range(len(self)):
            _, mask = self[idx]
            all_labels.extend(mask.numpy().flatten())
        
        class_distribution = Counter(all_labels)
        total_pixels = sum(class_distribution.values())
        
        print("\nClass Distribution:")
        for class_name, (r, g, b) in self.class_dict.items():
            class_id = list(self.class_dict.values()).index((r, g, b))
            count = class_distribution[class_id]
            percentage = (count / total_pixels) * 100
            print(f"{class_name}: {count} pixels ({percentage:.2f}%)")

        # Calculate mean and std of image intensities
        all_images = np.stack([self[idx][0].numpy() for idx in range(len(self))])
        mean = np.mean(all_images, axis=(0, 2, 3))
        std = np.std(all_images, axis=(0, 2, 3))
        
        print(f"\nImage Statistics:")
        print(f"Mean pixel intensity: {mean}")
        print(f"Std pixel intensity: {std}")
