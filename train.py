import torch
import wandb
from torch.utils.data import DataLoader
from models.ssd import get_ssd_model
from models.faster_rcnn import get_faster_rcnn_model
from utils.dataset import COCODataset, collate_fn
from torchvision.ops import box_iou
import torch.nn.utils as nn_utils

# Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 0.001
MODEL_TYPE = 'faster_rcnn'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SAMPLE_SIZE = 500
CLIP_VALUE = 2.0

print("Initializing Wandb...")
wandb.init(project='deep-learning-image-analysis')
print("Wandb initialized.")

print("Loading model...")
if MODEL_TYPE == 'ssd':
    model = get_ssd_model()
else:
    model = get_faster_rcnn_model()
model.to(DEVICE)
print("Model loaded.")

print("Loading data...")
train_dataset = COCODataset(split='train', sample_size=SAMPLE_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
val_dataset = COCODataset(split='val', sample_size=SAMPLE_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
print("Data loaded.")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def evaluate(model, data_loader, device):
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            for target, output in zip(targets, outputs):
                num_pred_boxes = len(output['boxes'])
                num_target_boxes = len(target['boxes'])
                if num_pred_boxes == 0 or num_target_boxes == 0:
                    continue
                iou = box_iou(output['boxes'].cpu(), target['boxes'].cpu())
                iou_scores.append(iou.diagonal().mean().item())
    
    model.train()
    if len(iou_scores) == 0:
        print("No valid IoU scores found.")
        return 0.0
    return sum(iou_scores) / len(iou_scores)

print("Starting training...")
model.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for i, (images, targets) in enumerate(train_loader):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        nn_utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)
        optimizer.step()

        epoch_loss += losses.item()
        if (i + 1) % 10 == 0:
            print(f'Batch {i + 1}/{len(train_loader)}, Loss: {losses.item()}')

        wandb.log({'loss': losses.item()})

    avg_loss = epoch_loss / len(train_loader)
    val_iou = evaluate(model, val_loader, DEVICE)
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss}, Validation IoU: {val_iou}')
    wandb.log({'epoch_loss': avg_loss, 'val_iou': val_iou})

    scheduler.step()

torch.save(model.state_dict(), 'model.pth')
print("Training completed and model saved.")
