import torch
import torchvision.models as models
import torch.nn as nn

class ViolenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Pretrained MobileNetV2 as feature extractor
        mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # LSTM for temporal reasoning across frames
        self.lstm = nn.LSTM(input_size=1280, hidden_size=256, 
                           num_layers=2, batch_first=True)
        self.classifier = nn.Linear(256, 2)  # violent / non-violent
    
    def forward(self, x):
        # x shape: (batch, frames, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = x.view(B, T, -1)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])

model = ViolenceClassifier()
# Save architecture
torch.save(model.state_dict(), 'violence_classifier.pt')
print("Model created. Now we need to load pretrained violence weights.")