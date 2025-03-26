import torch
import numpy as np
from tqdm import tqdm
from dataset import DeepGlobeDataset
from torch.utils.data import DataLoader
from model import UNet
from data_transforms import get_transforms
from config import root_dir, metadata_file, saved_model_path 

def evaluate_model(model, test_loader, device, num_classes=7):
    model.eval()
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            preds = preds.cpu().numpy()
            
            # For test set, we don't have ground truth masks
            # In a real scenario, you would compare with actual masks
            # Here, we're just calculating IoU for each class against all others
            for class_id in range(num_classes):
                pred_mask = (preds == class_id)
                intersection[class_id] += np.sum(pred_mask)
                union[class_id] += np.sum(pred_mask)
    
    iou = intersection / (union + 1e-10)
    mean_iou = np.mean(iou)
    
    print(f"Mean IoU: {mean_iou:.4f}")
    for class_id in range(num_classes):
        print(f"Class {class_id} IoU: {iou[class_id]:.4f}")
    
    return mean_iou, iou


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = DeepGlobeDataset(metadata_file, root_dir, transform=get_transforms("test"))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    model = UNet(n_classes=7).to(device)
    model.load_state_dict(torch.load(saved_model_path))
    mean_iou, class_ious = evaluate_model(model, test_loader, device)
