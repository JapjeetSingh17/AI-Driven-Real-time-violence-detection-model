import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class ViolenceDetectionModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ViolenceDetectionModel, self).__init__()
        
        # Load pre-trained ResNet3D (r3d_18)
        if pretrained:
            weights = R3D_18_Weights.DEFAULT
            self.model = r3d_18(weights=weights)
        else:
            self.model = r3d_18(weights=None)
            
        # Modify the final classification layer (fc)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        """Freezes all layers except the final classification head."""
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze the fc layer
        for param in self.model.fc.parameters():
            param.requires_grad = True
            
    def unfreeze_all(self):
        """Unfreezes all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

if __name__ == "__main__":
    # Test model definition
    model = ViolenceDetectionModel(num_classes=2)
    sample_input = torch.randn(1, 3, 16, 112, 112) # (B, C, T, H, W)
    output = model(sample_input)
    print(f"Model output shape: {output.shape}")
