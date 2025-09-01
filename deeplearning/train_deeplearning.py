import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from deeplearning.model_generator_deeplearning import EfficientNetYOLO
from deeplearning.yolo_target_generator_deeplearning import build_yolo_targets_batch
from deeplearning.dataset_deeplearning import YOLODataset

seed = 0
torch.manual_seed(seed)

# Hyperparameters
num_classes = 4
anchors = 3
lr = 1e-4
epochs = 100
batch_size = 64
grid_size = 7  # assuming final feature map is 7x7 for 224x224 input
anchors = torch.tensor([[0.1,0.1], [0.2,0.2], [0.4,0.4]])
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# write a collate function to handle multiple boxes per image
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    targets = [t for t in targets]
    max_boxes = max(t.shape[0] for t in targets)
    padded_targets = torch.zeros((len(targets), max_boxes, 5))
    for i, t in enumerate(targets):
        if t.shape[0] > 0:
            padded_targets[i, :t.shape[0], :] = t
    return images, padded_targets


dataset = YOLODataset("/Volumes/Crucial X6/object detection/FlyObjDataset/train", img_size=224)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize model
model = EfficientNetYOLO(num_classes=num_classes, anchors=len(anchors)).to(device)

# Loss functions
mse_loss = nn.MSELoss(reduction='none')
cce_loss = nn.CrossEntropyLoss(reduction='none')

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # shape: (B, anchors, 5 + n_classes, H, W)

        # use yolo target generator to convert targets to shape (B, anchors, 5, grid_size, grid_size)
        targets = build_yolo_targets_batch(targets, anchors, grid_size, num_classes, device=device).to(device)
        target_objectness = targets[:, :, 0, :, :].unsqueeze(2)

        pred_cls = outputs[:, :, 5:, :, :]
        target_cls = targets[:, :, 5:, :, :]

        #convert to idx from one hot
        target_cls_idx = target_cls.argmax(dim=2)
        pred_box = outputs[:, :, 1:5, :, :]
        target_box = targets[:, :, 1:5, :, :]

        # Compute losses without reduction
        loss_box = mse_loss(pred_box, target_box) * target_objectness
        loss_box_term = loss_box.sum() / (target_objectness.sum() + 1e-6)

        loss_cls = cce_loss(torch.swapaxes(pred_cls, 1, 2), target_cls_idx)
        loss_cls = torch.where(target_objectness.squeeze(2) > 0, loss_cls, torch.tensor(0.0, device=device))
        loss_cls_term = loss_cls.sum() / (target_objectness.sum() + 1e-6)

        loss_obj_term = mse_loss(outputs[:, :, 0, :, :], target_objectness.squeeze(2)).mean()
        # Combine losses
        loss = 0.1*loss_box_term + loss_cls_term + 5*loss_obj_term

        loss.backward()
        optimizer.step()
        print(f'Losses - Box: {loss_box_term.item():.4f}, Cls: {loss_cls_term.item():.4f}, Obj: {loss_obj_term.item():.4f}, Total: {loss.item():.4f}')
        running_loss += loss.item()





    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")

print("Training done!")
