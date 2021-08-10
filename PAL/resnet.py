from torchvision import transforms
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor():
    def __init__(self):
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)
        for p in self.resnet18.parameters():
            p.requires_grad = False
        self.transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                )
        self.flatten = nn.Flatten()


    def extract_features(self, img):
        feat = self.resnet18(self.transform(img).unsqueeze(0))
        feat = self.flatten(feat).squeeze(0)
        return feat
