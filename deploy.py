import torch
import pytorch_lightning as pl
from torch import nn
from torchvision import models
from torchvision import transforms
from PIL import Image

# Assuming ResNetModel is defined as before
class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

def load_clf_model(ckpt_path):
    """Load the classifier model."""

    # To deploy a checkpoint and use for inference
    trained_model = ResNetModel.load_from_checkpoint(
        ckpt_path
    )  # , map_location=torch.device("cpu"))

    # # move the model to the GPU
    # trained_model.to("cuda")

    # turn off the training mode
    trained_model.eval()

    return trained_model


def predict_batch(pil_images, model):
    """
    Predict the confidence scores for a batch of PIL images.

    Parameters:
    - pil_images (list of PIL.Image.Image): List of input PIL Image objects.
    - model (torch.nn.Module): Trained model.

    Returns:
    - list of float: List of confidence scores for the class label `1` for each image.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Transform each image and stack them into a batch
    batch = torch.stack([transform(image.convert("RGB")) for image in pil_images])

    # # Move the batch to the GPU
    # batch = batch.to("cpu")

    with torch.no_grad():  # No need to compute gradients for inference
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)

        # prob shape is [44, 1, 3]
        peripheral_confidence_scores = probs[:, 1].cpu().numpy()
        clot_confidence_scores = probs[:, 2].cpu().numpy()
        adequate_confidence_scores = probs[:, 0].cpu().numpy()

    return (
        peripheral_confidence_scores,
        clot_confidence_scores,
        adequate_confidence_scores,
    )

def run_one_image(model_path, image_path):
    """ Run the model on a single image. 
    Print the final output as a dictionary.
    e.g. {"peripheral": 0.9, "clot": 0.1, "adequate": 0.0}
    """
    
    # Load the model
    model = load_clf_model(model_path)

    # Load the image
    image = Image.open(image_path)

    # make sure to downsample the image by a factor of 8
    downsample_factor = 8
    size = (512 // downsample_factor, 512 // downsample_factor)
    image = transforms.functional.resize(image, size)

    # Predict
    peripheral_confidence_scores, clot_confidence_scores, adequate_confidence_scores = predict_batch(
        [image], model
    )

    # Print the final output
    print(
        {
            "peripheral": peripheral_confidence_scores[0],
            "clot": clot_confidence_scores[0],
            "adequate": adequate_confidence_scores[0],
        }
    )

    return {
        "peripheral": peripheral_confidence_scores[0],
        "clot": clot_confidence_scores[0],
        "adequate": adequate_confidence_scores[0],
    }

if __name__ == "__main__":
    run_one_image("/Users/neo/Documents/Research/MODELS/2024-02-01 Region Clf/8/version_0/checkpoints/epoch=99-step=18300.ckpt", 
                  "/Users/neo/Documents/Research/DeepHeme/LLData/bma_region_clf_data_iter_2/selected_focus_regions/228.jpg")