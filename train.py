import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
from model import UNet
from lovasz_loss import lovasz_softmax
from dataset import DeepGlobeDataset
import os
from data_transforms import get_transforms
from config import root_dir, metadata_file, saved_model_path, BATCH_SIZE, NUM_EPOCHS, NUM_CLASSES


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir):
    swa_model = AveragedModel(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    swa_start = 100
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), saved_model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

    return swa_model, best_val_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = UNet(n_classes=NUM_CLASSES).to(device)
    criterion = lovasz_softmax
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Loading images..")
    train_dataset = DeepGlobeDataset(metadata_file, root_dir, transform=get_transforms("train"))
    val_dataset = DeepGlobeDataset(metadata_file, root_dir, transform=get_transforms("val"))
    
    # Print stats for train and validation datasets
    train_dataset.print_stats()
    val_dataset.print_stats()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


    print("Starting model training..")
    trained_model, best_val_loss = train_model(model, train_loader, val_loader, 
                                               criterion, optimizer, num_epochs=NUM_EPOCHS, 
                                               device=device, save_dir="/output")
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    print("Model saved at /output/best_model.pth")