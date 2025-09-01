import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class YOLODataset(Dataset):
    def __init__(self, root_dir, img_size=224, transform=None):
        """
        Args:
            root_dir (str): Path to dataset root containing 'images' and 'labels' folders.
            img_size (int): Resize images to this size.
            transform (torchvision.transforms): Optional image transformations.
        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "images"
        self.label_dir = self.root_dir / "labels"
        self.img_paths = sorted(list(self.img_dir.glob("*.jpg")))

        #remove all images beginning with ._
        self.img_paths = [p for p in self.img_paths if not p.name.startswith("._")]
        self.img_size = img_size
        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Load labels
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    # Only consider lines with 5 numbers (YOLO bbox format)
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        boxes.append([cls, x, y, w, h])

        boxes = torch.tensor(boxes, dtype=torch.float32)  # (num_boxes, 5)

        return img, boxes


# Example usage
if __name__ == "__main__":
    dataset = YOLODataset("/Volumes/Crucial X6/object detection/FlyObjDataset/train")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
