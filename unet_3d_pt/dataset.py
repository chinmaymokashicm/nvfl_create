import os

import nibabel as nib
from torchvision import transforms
from torch.utils.data import Dataset

class PTDataset(Dataset):
    def __init__(self, data_dir="dataset", is_train=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train= is_train
        self._load_filepaths()
        
    # Load filepaths
    def _load_filepaths(self):
        foldername = self.data_dir
        subfolder = "train" if self.is_train else "test"
        self.filepath_images = sorted([os.path.join(foldername, subfolder, "images", f) for f in os.listdir(os.path.join(foldername, subfolder, "images")) if f.endswith(".nii.gz")])
        self.filepath_labels = sorted([os.path.join(foldername, subfolder, "labels", f) for f in os.listdir(os.path.join(foldername, subfolder, "labels")) if f.endswith(".nii.gz")])

    # Load image/label file
    def _load_data(self, filepath_image):
        img = nib.load(filepath_image)
        data = img.get_fdata()
        return data

    # Return length of dataset
    def __len__(self):
        return len(self.filepath_images)

    # Get image and label by index
    def __getitem__(self, idx):
        image, label = self._load_data(self.filepath_images[idx]), self._load_data(self.filepath_labels[idx])
        
        if self.transform:
            image, label = self.transform(image), self.transform(label)
        
        return image, label

