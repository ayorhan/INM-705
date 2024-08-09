import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from models.ssd import get_ssd_model
from models.faster_rcnn import get_faster_rcnn_model

# Load model
MODEL_TYPE = 'ssd'  # Change to 'faster_rcnn' for the complex model
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("Loading model...")
if MODEL_TYPE == 'ssd':
    model = get_ssd_model()
else:
    model = get_faster_rcnn_model()
model.load_state_dict(torch.load('model.pth'))
model.to(DEVICE)
model.eval()
print("Model loaded.")

# Load image
image_path = 'path_to_image.jpg'
image = Image.open(image_path).convert('RGB')
transform = T.Compose([T.ToTensor()])
image = transform(image).unsqueeze(0).to(DEVICE)

# Inference
print("Running inference...")
with torch.no_grad():
    prediction = model(image)
print("Inference completed.")

# Visualize
plt.imshow(image[0].permute(1, 2, 0).cpu())
for box in prediction[0]['boxes'].cpu():
    plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor='red', linewidth=2))
plt.show()
