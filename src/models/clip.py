from transformers import CLIPProcessor, CLIPModel
from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device
print(device)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
