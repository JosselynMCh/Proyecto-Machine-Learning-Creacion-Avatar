import pandas as pd
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CelebADataset(Dataset):
    """
    A custom dataset loader for the CelebA dataset.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Example usage of the dataset loader
if __name__ == '__main__':
    dataset = CelebADataset(root_dir='path/to/celeba', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)