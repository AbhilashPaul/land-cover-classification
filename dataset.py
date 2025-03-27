import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class DeepGlobeDataset(Dataset):
    def __init__(self, metadata_file, root_dir, transform=None):
        """
        Args:
            metadata_file (string): Path to the CSV file with image paths.
            root_dir (string): Root directory containing 'train', 'val', and 'test' folders.
            transform (callable, optional): Transformations to be applied to images and masks.
        """
        self.metadata = pd.read_csv(metadata_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Class definitions with RGB values and class indices
        self.class_dict = {
            'urban_land': (0, 255, 255),
            'agriculture_land': (255, 255, 0),
            'rangeland': (255, 0, 255),
            'forest_land': (0, 255, 0),
            'water': (0, 0, 255),
            'barren_land': (255, 255, 255),
            'unknown': (0, 0, 0)
        }
        
        # Precompute RGB to class index mapping for faster conversion
        self.color_to_index = np.zeros((256, 256, 256), dtype=np.uint8)
        for idx, rgb in enumerate(self.class_dict.values()):
            self.color_to_index[rgb] = idx

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get image and mask paths from metadata
        row = self.metadata.iloc[idx]
        image_path = os.path.join(self.root_dir, row['sat_image_path'])
        
        # Open satellite image
        image = Image.open(image_path).convert("RGB")
        
        # Open mask if available (for train/val splits)
        if 'mask_path' in row and pd.notna(row['mask_path']):
            mask_path = os.path.join(self.root_dir, row['mask_path'])
            mask = Image.open(mask_path).convert("RGB")
        else:
            mask = Image.new('RGB', image.size, (0, 0, 0))  # Create blank mask for test set
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Convert RGB mask to class indices
        mask = self._rgb_to_index(mask)

        return image, mask

    def _rgb_to_index(self, mask):
        """Convert RGB mask tensor to class index tensor."""
        if isinstance(mask, Image.Image):
            mask = np.array(mask)  # Convert PIL Image to NumPy array
        
        if isinstance(mask, torch.Tensor):
            mask = mask.permute(1, 2, 0).numpy().astype(np.uint8)  # Convert Tensor to NumPy array
        
        index_mask = self.color_to_index[
            mask[..., 0], mask[..., 1], mask[..., 2]  # Red channel | Green channel | Blue channel
        ]
        
        return torch.from_numpy(index_mask).long()
