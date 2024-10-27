import os
import requests
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


class PlantDatasetCollector:
    def __init__(self, base_path="./plant_dataset"):
        self.base_path = base_path
        self.image_size = (224, 224)  # Standard size for many CNN models
        os.makedirs(base_path, exist_ok=True)

    def download_inat_data(self, api_key, taxon_id):
        """
        Download data from iNaturalist API
        taxon_id: ID for plants (47126)
        """
        url = f"https://api.inaturalist.org/v1/observations"
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": True,
            "per_page": 200,
            "order": "desc",
            "order_by": "created_at"
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        response = requests.get(url, params=params, headers=headers)
        return response.json()

    def process_image(self, image_path):
        """Process and normalize images for model training"""
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        image = Image.open(image_path).convert('RGB')
        return transform(image)

    def create_dataset_structure(self, data_info):
        """
        Create organized dataset structure:
        dataset/
        ├── train/
        │   ├── species1/
        │   ├── species2/
        ├── val/
        │   ├── species1/
        │   ├── species2/
        └── test/
            ├── species1/
            ├── species2/
        """
        splits = ['train', 'val', 'test']
        for split in splits:
            for species in data_info['species'].unique():
                os.makedirs(f"{self.base_path}/{split}/{species}", exist_ok=True)

    def create_metadata_file(self, data_info):
        """Create metadata CSV with image paths and labels"""
        metadata = {
            'image_path': [],
            'species': [],
            'source': [],
            'split': []
        }
        # Add dataset details
        return pd.DataFrame(metadata)


def main():
    # Initialize collector
    collector = PlantDatasetCollector()

    # Example sources and their configurations
    dataset_sources = {
        'plantnet': {
            'api_url': 'https://my-api.plantnet.org/v2/identify',
            'api_key': 'your_plantnet_api_key'
        },
        'inaturalist': {
            'api_url': 'https://api.inaturalist.org/v1',
            'api_key': 'your_inat_api_key'
        }
    }

    # Dataset specifications
    dataset_specs = {
        'min_images_per_class': 100,
        'max_images_per_class': 1000,
        'min_species': 50,
        'image_quality': {
            'min_resolution': (800, 800),
            'max_size_mb': 5
        }
    }

    # Split ratios
    split_ratios = {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15
    }

    collector.create_dataset_structure()
    collector.download_inat_data()

if __name__ == "__main__":
    main()