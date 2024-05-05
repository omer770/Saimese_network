import torch
import torchvision.models as models

class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)  # Using a pre-trained ResNet
        for param in self.feature_extractor.parameters(): param.requires_grad = False
        self.feature_extractor.fc = torch.nn.Identity()
        self.comparator = torch.nn.Sequential(
                      torch.nn.Linear(512 * 2, 512),
                      torch.nn.ReLU(),
                      torch.nn.Dropout(0.5), 
                      torch.nn.Linear(512, 256),
                      torch.nn.ReLU(),
                      torch.nn.Linear(256, 5)) 

    def forward(self, image1, image2):
        features1 = self.feature_extractor(image1)
        features2 = self.feature_extractor(image2)
        combined_features = torch.cat((features1, features2), dim=1)
        return self.comparator(combined_features) 
