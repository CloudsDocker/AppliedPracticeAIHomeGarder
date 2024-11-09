import base64
import requests
import json
from PIL import Image
import os

# Usage example
API_KEY = "..........."  # Replace with your actual API key
image_path = "any_tomato.jpeg"  # Replace with your image path
# image_path = "/Users/todd.zhang/Downloads/Monstera-Problems.jpg"
# image_path = "/Users/todd.zhang/Downloads/pests-and-diseases-roses-520-500.jpeg"

class PlantIdentifier:
    def __init__(self, api_key):
        """Initialize with your Plant.id API key"""
        self.api_key = api_key
        self.api_endpoint = "https://plant.id/api/v2/identify"

    def encode_image(self, image_path):
        """Encode image to base64 string"""
        with open(image_path, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')

    def preprocess_image(self, image_path):
        """Basic image preprocessing"""
        # Open and resize image if needed
        with Image.open(image_path) as img:
            # Convert to RGB if image is in a different mode
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize if image is too large
            max_size = 2000
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple([int(x * ratio) for x in img.size])
                img = img.resize(new_size, Image.Refilter.LANCZOS)

            # Save preprocessed image
            preprocessed_path = 'preprocessed_' + os.path.basename(image_path)
            img.save(preprocessed_path, 'JPEG', quality=95)
            return preprocessed_path

    def identify_plant(self, image_path):
        """Identify plant from image"""
        # Preprocess the image
        processed_image_path = self.preprocess_image(image_path)

        # Prepare the request
        encoded_image = self.encode_image(processed_image_path)

        # Request data
        data = {
            "images": [encoded_image],
            "modifiers": ["similar_images"],
            "plant_details": [
                "common_names",
                "taxonomy",
                "url",
                "wiki_description",
                "edible_parts",
                "cultivation"
            ],
            "disease_details": [
                "common_names",
                "description",
                "treatment"
            ]
        }

        # Request headers
        headers = {
            "Content-Type": "application/json",
            "Api-Key": self.api_key
        }

        # Make the request
        response = requests.post(
            self.api_endpoint,
            json=data,
            headers=headers
        )

        # Clean up preprocessed image
        os.remove(processed_image_path)

        return self.process_response(response.json())

    def process_response(self, response):
        """Process and format the API response"""
        if not response.get('suggestions'):
            return {
                'success': False,
                'error': 'No plants identified'
            }

        # Get the best match
        best_match = response['suggestions'][0]

        result = {
            'success': True,
            'plant_details': {
                'name': best_match.get('plant_name', 'Unknown'),
                'probability': f"{best_match.get('probability', 0) * 100:.1f}%",
                'common_names': best_match.get('plant_details', {}).get('common_names', []),
                'scientific_name': best_match.get('plant_details', {}).get('scientific_name', ''),
                'taxonomy': best_match.get('plant_details', {}).get('taxonomy', {}),
                'description': best_match.get('plant_details', {}).get('wiki_description', {}).get('value', ''),
                'edible_parts': best_match.get('plant_details', {}).get('edible_parts', []),
                'cultivation': best_match.get('plant_details', {}).get('cultivation', '')
            }
        }

        # Add disease information if available
        if 'health_assessment' in response:
            health = response['health_assessment']
            result['health_assessment'] = {
                'is_healthy': health.get('is_healthy', True),
                'diseases': [{
                    'name': disease.get('name', ''),
                    'probability': f"{disease.get('probability', 0) * 100:.1f}%",
                    'description': disease.get('description', ''),
                    'treatment': disease.get('treatment', '')
                } for disease in health.get('diseases', [])]
            }

        return result


def main():

    # Initialize identifier
    identifier = PlantIdentifier(API_KEY)

    try:
        # Get identification results
        result = identifier.identify_plant(image_path)

        if result['success']:
            # Print basic information
            print("\n🌱 Plant Identification Results:")
            print(f"\nPlant Name: {result['plant_details']['name']}")
            print(f"Confidence: {result['plant_details']['probability']}")
            print(f"\nCommon Names: {', '.join(result['plant_details']['common_names'])}")

            # Print description
            if result['plant_details']['description']:
                print("\n📝 Description:")
                print(result['plant_details']['description'][:500] + "...")

            # Print cultivation tips
            if result['plant_details']['cultivation']:
                print("\n🌿 Cultivation Tips:")
                print(result['plant_details']['cultivation'])

            # Print health assessment if available
            if 'health_assessment' in result:
                print("\n🏥 Health Assessment:")
                if result['health_assessment']['is_healthy']:
                    print("Plant appears healthy!")
                else:
                    print("Potential health issues detected:")
                    for disease in result['health_assessment']['diseases']:
                        print(f"\nIssue: {disease['name']}")
                        print(f"Probability: {disease['probability']}")
                        if disease['treatment']:
                            print(f"Treatment: {disease['treatment']}")
        else:
            print(f"Error: {result['error']}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()