import requests
import base64
from pathlib import Path
import json


def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_image_stream(image_path):
    """Analyze image using Ollama API with streaming response"""
    url = "http://localhost:11434/api/generate"  # Changed back to /api/generate

    try:
        # Verify image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Encode image
        base64_image = encode_image_to_base64(image_path)

        # Prepare the request
        payload = {
            "model": "llava-llama3",
            "prompt": """Please analyze this plant image and provide:
                1. Plant identification
                2. Current health status
                3. Any visible problems or diseases
                4. Treatment recommendations if needed
                5. General care tips""",
            "images": [base64_image],
            "stream": True
        }

        # Make the request and stream the response
        print("\n🌿 Plant Analysis Results:")
        print("-" * 50)

        full_response = ""
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    chunk = json_response.get('response', '')
                    print(chunk, end='', flush=True)
                    full_response += chunk

        return full_response

    except requests.exceptions.ConnectionError:
        print("Connection Error: Could not connect to Ollama server")
        print("\nPlease check:")
        print("1. Is Ollama running? Run 'ps aux | grep ollama' to check")
        print("2. Is it running on port 11434?")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check if llava model is installed: ollama list")
        print("2. If not installed, run: ollama pull llava")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check if Ollama is running: ps aux | grep ollama")
        print("2. Verify llava model is installed: ollama list")
        print("3. Check the image path")
        return None


def main():
    img_path = '/Users/todd.zhang/Downloads/monash_tomato.jpeg'

    # First, verify Ollama server is responsive
    try:
        health_check = requests.get("http://localhost:11434/api/version")
        health_check.raise_for_status()
        print("✅ Ollama server is running")

        # Check installed models
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("\n📋 Installed models:")
            print(models)
    except Exception as e:
        print("❌ Ollama server is not responding")
        print("Please start Ollama and try again")
        return

    # Proceed with analysis
    analysis = analyze_image_stream(img_path)

    if analysis:
        # Save the analysis to a file
        with open('plant_analysis.txt', 'w') as f:
            f.write(analysis)
        print("\n\n✅ Analysis saved to plant_analysis.txt")


if __name__ == "__main__":
    main()