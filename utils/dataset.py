import deeplake
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Augmentation pipeline using Albumentations
tform_train = A.Compose([
    A.RandomSizedBBoxSafeCrop(width=128, height=128, erosion_rate=0.2),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'bbox_ids'], min_area=25, min_visibility=0.6))

# Conversion script for bounding boxes from coco to Pascal VOC format
def coco_2_pascal(boxes):
    return np.stack((boxes[:, 0], boxes[:, 1], boxes[:, 0] + np.clip(boxes[:, 2], 1, None), boxes[:, 1] + np.clip(boxes[:, 3], 1, None)), axis=1)

# Transformation function for pre-processing the deeplake sample before sending it to the model
def transform(sample_in):
    # Convert boxes to Pascal VOC format
    boxes = coco_2_pascal(sample_in['boxes'].numpy())

    # Convert any grayscale images to RGB
    images = sample_in['images'].numpy()
    if images.shape[2] == 1:
        images = np.repeat(images, int(3/images.shape[2]), axis=2)

    # Pass all data to the Albumentations transformation
    transformed = tform_train(image=images,
                              bboxes=boxes,
                              bbox_ids=np.arange(boxes.shape[0]),
                              class_labels=sample_in['categories'].numpy())

    # Convert boxes and labels from lists to torch tensors
    labels_torch = torch.tensor(transformed['class_labels'], dtype=torch.int64)

    boxes_torch = torch.zeros((len(transformed['bboxes']), 4), dtype=torch.float32)
    for b, box in enumerate(transformed['bboxes']):
        boxes_torch[b, :] = torch.tensor(np.round(box), dtype=torch.float32)

    target = {'labels': labels_torch, 'boxes': boxes_torch}

    return transformed['image'], target

# Custom dataset class
class COCODataset(Dataset):
    def __init__(self, split='train', sample_size=100):
        print("Loading dataset...")
        self.ds = deeplake.load('hub://activeloop/coco-train') if split == 'train' else deeplake.load('hub://activeloop/coco-val')
        self.transform = transform
        self.sample_size = sample_size
        print("Dataset loaded.")

    def __len__(self):
        return min(len(self.ds), self.sample_size)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        image, target = self.transform(sample)
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))
