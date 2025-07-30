import torch
import jax
from eqxvision.models.classification.resnet import resnet50
from eqxvision.utils import CLASSIFICATION_URLS

model = resnet50(torch_weights=CLASSIFICATION_URLS["resnet50"])

print("✅ ResNet-50 model loaded successfully with PyTorch weights!")
print(f"Model type: {type(model)}")
print("Ready for inference!")