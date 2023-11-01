import torch
import torch.nn as nn
import torchvision.models as models


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, model_path):
        super(VGG16FeatureExtractor, self).__init__()
        # Load VGG16 model
        original_model = models.vgg16(pretrained=False)
        # Replacement of the classifier part to fit the binary classification task
        num_features = original_model.classifier[6].in_features
        original_model.classifier[6] = nn.Linear(num_features, 2)
        # load trained weight
        original_model.load_state_dict(torch.load(model_path))
        # Use only the feature extraction part of VGG and acquire features at the desired layer
        self.features = nn.Sequential(
            # Get the output of the first 9 layers
            *list(original_model.features.children())[:9]
        )

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {1, 3, 5, 8}:  # 2nd,4th,6th and 9th
                results.append(x)
        return results
