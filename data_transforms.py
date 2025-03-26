from torchvision import transforms

def get_transforms(phase: str):
    if phase == "train":
        return transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.ToTensor(),
        ])