# Data loading
root_dir = "dataset"
metadata_file = "dataset/metadata.csv"
saved_model_path = "output/best_model.pth"

IMAGE_SIZE = (256, 256)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 4
NUM_EPOCHS = 5
NUM_CLASSES = 7
