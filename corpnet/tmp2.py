import tensorflow as tf
import numpy as np
import os
import json
from pathlib import Path


class ModelInspector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = tf.saved_model.load(model_path)

    def inspect_model_structure(self):
        """Inspect the model's structure and files"""
        print("\n📂 Model Directory Contents:")
        for root, dirs, files in os.walk(self.model_path):
            level = root.replace(self.model_path, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

    def check_assets_for_labels(self):
        """Check assets directory for label files"""
        print("\n📑 Checking assets for label files...")
        assets_path = os.path.join(self.model_path, 'assets')
        if os.path.exists(assets_path):
            for file in os.listdir(assets_path):
                if 'label' in file.lower() or 'class' in file.lower():
                    print(f"Found potential label file: {file}")
                    with open(os.path.join(assets_path, file), 'r') as f:
                        print(f.read())

    def inspect_model_outputs(self):
        """Inspect model outputs to determine number of classes"""
        print("\n🔍 Inspecting model outputs...")

        # Create a dummy input (adjust size if needed)
        dummy_input = tf.zeros([1, 224, 224, 3])

        try:
            # Make prediction
            result = self.model(dummy_input, training=False)
            print(f"Output shape: {result.shape}")
            print(f"Number of classes: {result.shape[-1]}")
            return result.shape[-1]
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None

    def check_saved_model_proto(self):
        """Try to extract information from saved_model.pb"""
        print("\n📄 Checking saved_model.pb...")
        try:
            signatures = self.model.signatures
            print("Available signatures:", list(signatures.keys()))

            # Try to get output details
            if 'serving_default' in signatures:
                serving_signature = signatures['serving_default']
                print("\nServing signature outputs:", serving_signature.structured_outputs)
        except Exception as e:
            print(f"Error reading signatures: {str(e)}")


def main():
    # Update this path to your model location
    model_path = "models/feature-vector-concat"  # Update this to your model path

    print("🔎 Starting Model Inspection")
    inspector = ModelInspector(model_path)

    # Run all inspections
    inspector.inspect_model_structure()
    inspector.check_assets_for_labels()
    num_classes = inspector.inspect_model_outputs()
    inspector.check_saved_model_proto()

    # If we found the number of classes, create a temporary mapping
    if num_classes:
        print("\n📝 Creating temporary class mapping...")
        # These are example classes - update based on actual model documentation
        # or through testing with known images
        temp_class_names = {
            0: "Apple Scab",
            1: "Apple Black Rot",
            2: "Apple Cedar Rust",
            3: "Apple Healthy",
            4: "Background No Plant",
            5: "Blueberry Healthy",
            6: "Cherry Healthy",
            7: "Cherry Powdery Mildew",
            8: "Corn Cercospora Leaf Spot",
            9: "Corn Common Rust",
            10: "Corn Northern Leaf Blight",
            11: "Corn Healthy",
            # Add more based on actual number of classes
        }

        print(f"\nFound {num_classes} classes.")
        print("\nPotential class mapping (verify with documentation):")
        for i in range(min(num_classes, len(temp_class_names))):
            print(f"Class {i}: {temp_class_names.get(i, f'Unknown Class {i}')}")

        print("\n⚠️ Note: This class mapping is tentative and should be verified with:")
        print("1. Model documentation")
        print("2. Testing with known images")
        print("3. Kaggle model page information")
        print("4. Original research paper")


def test_with_known_image(model_path, image_path):
    """
    Test the model with a known image to help identify classes
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    import numpy as np

    def preprocess_image(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize
        return x

    # Load model
    model = tf.saved_model.load(model_path)

    # Preprocess image
    img = preprocess_image(image_path)

    # Make prediction
    pred = model(img, training=False)

    # Get prediction details
    pred_index = tf.argmax(pred, axis=1).numpy()[0]
    confidence = tf.nn.softmax(pred, axis=1).numpy()[0][pred_index]

    print(f"\nPrediction for image: {image_path}")
    print(f"Predicted class index: {pred_index}")
    print(f"Confidence: {confidence:.2%}")

    return pred_index, confidence


if __name__ == "__main__":
    main()

    # Uncomment and use this to test with known images
    # test_with_known_image("models/feature-vector-concat", "path_to_known_image.jpg")