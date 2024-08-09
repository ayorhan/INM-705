import torchvision

def get_ssd_model():
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.num_classes = 91  # COCO has 91 classes
    return model
