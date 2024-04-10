from train import ResNetModel
import torch
from torchvision import transforms
from PIL import Image


def load_model_checkpoint(checkpoint_path):
    """
    Load a model checkpoint and return the model object.

    Parameters:
    - checkpoint_path: str, path to the model checkpoint.

    Returns:
    - model: PyTorch model loaded with checkpoint weights.
    """
    # Assuming ResNetModel is defined elsewhere as in your provided code
    model = ResNetModel(
        num_classes=2
    )  # num_classes should match your training configuration
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')["state_dict"])

    model.eval()  # Set the model to evaluation mode

    return model


def predict_image(model, image_path):
    """
    Takes a model object and an image path, preprocesses the image, and returns the classification confidence score.

    Parameters:
    - model: The model object for prediction.
    - image_path: str, path to the image file.

    Returns:
    - confidence_score: The confidence score of the classification.
    """
    # Image preprocessing
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (512, 512)
            ),  # Assuming you want to keep the original size used in training
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # Inference without tracking gradients
        outputs = model(image)
        # Assuming binary classification with softmax at the end
        confidence_score = torch.softmax(outputs, dim=1).numpy()[0]

    return float(confidence_score[0])


if __name__ == "__main__":
    ckpt_path = "/Users/neo/Documents/Research/MODELS/2024-04-06 BMARegionClf 1000Epochs/lightning_logs/1/version_0/checkpoints/epoch=999-step=55000.ckpt"
    model = load_model_checkpoint(ckpt_path)
    image_path = "/Users/neo/Documents/Research/MODELS/results/results_bma_aml_v1_LITE/H21-8692;S10;MSKB - 2024-01-02 22.54.24/focus_regions/high_mag_unannotated/12718.jpg"
    confidence_score = predict_image(model, image_path)
    print(f"Confidence score: {confidence_score}")
