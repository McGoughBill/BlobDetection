import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetYOLO(nn.Module):
    def __init__(self, num_classes=80, anchors=3):
        """
        YOLO-style object detection with EfficientNet-B0 backbone.

        Args:
            num_classes (int): Number of object classes.
            anchors (int): Number of anchor boxes per grid cell.
        """
        super(EfficientNetYOLO, self).__init__()

        # Load EfficientNetV2-S backbone
        backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Remove the classifier and keep only feature extractor
        self.backbone = nn.Sequential(*list(backbone.features.children()))

        # YOLO-style head
        self.num_classes = num_classes
        self.anchors = anchors

        #make yolo head a bit deeper with residual connections
        self.yolo_head = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, anchors * (5 + num_classes), kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # Extract backbone features
        with torch.no_grad():
            features = self.backbone(x)

        # Apply YOLO head
        out = self.yolo_head(features)

        # Reshape to (batch, anchors, 5 + num_classes, H, W)
        batch_size, _, H, W = out.shape
        out = out.view(batch_size, self.anchors, 5 + self.num_classes, H, W)
        return out


# Example usage
if __name__ == "__main__":
    import torch
    import time
    print(torch.backends.mps.is_available())
    x = torch.randn(2, 3, 640, 640).to("mps" if torch.backends.mps.is_available() else "cpu")
    model = EfficientNetYOLO(num_classes=20, anchors=3).to("mps" if torch.backends.mps.is_available() else "cpu")

    start_time = time.time()
    for i in range(10):
        y = model(x)
    end_time = time.time()
    print(f"Time taken for 10 forward passes: {end_time - start_time:.4f} seconds")
    print("Output shape:", y.shape)  # Should be (2, 3, 25, H', W')
