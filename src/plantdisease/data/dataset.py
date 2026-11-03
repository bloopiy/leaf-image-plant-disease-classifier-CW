"""Dataset acquisition and schema validation."""

from datasets import ClassLabel, Dataset, Image, load_dataset

def validate_dataset_schema(dataset):
    """Validate the schema of the full dataset and return metadata."""

    if not isinstance(dataset, Dataset):
        raise TypeError("Expected a Hugging Face Dataset object.")

    features = dataset.features

    if "image" not in features or not isinstance(features["image"], Image):
        raise ValueError("Dataset must contain an Image feature named 'image'.")

    if "label" not in features or not isinstance(features["label"], ClassLabel):
        raise ValueError("Dataset must contain a ClassLabel feature named 'label'.")

    class_names = features["label"].names
    if not class_names:
        raise ValueError("Dataset contains no class labels.")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    return {
        "num_samples": len(dataset),
        "num_classes": len(class_names),
        "class_names": class_names,
        "image_column": "image",
        "label_column": "label",
    }

def load_plant_dataset(cfg):
    """Load and validate the full configured dataset as one Dataset."""
    dataset = load_dataset(cfg["data"]["dataset"]["id"], split="train")
    validate_dataset_schema(dataset)
    return dataset
