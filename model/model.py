import torch.nn as nn
from .functions import ReverseLayerF
from torchvision import models

class DANN_InceptionV3(nn.Module):
    def __init__(self):
        super(DANN_InceptionV3, self).__init__()
        model = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
        self.num_ftrs = model.fc.in_features
        self.class_classifier = nn.Linear(self.num_ftrs, 2)
        self.domain_classifier = nn.Linear(self.num_ftrs, 4)
        model.fc = nn.Identity()
        self.feature = model        
        
    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.shape[0], 3, 299, 299)
        try:
            feature = self.feature(input_data).logits
        except:
            feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output