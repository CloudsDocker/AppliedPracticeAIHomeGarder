import tensorflow as tf
import json
import os


def inspect_saved_model(model_path):
    """Inspect SavedModel contents"""
    print("\n1. Examining SavedModel contents:")
    print(os.listdir(model_path))

    
    assets_path = os.path.join(model_path, 'assets')
    if os.path.exists(assets_path):
        print("\nAssets directory contents:")
        print(os.listdir(assets_path))

        
        for file in os.listdir(assets_path):
            if 'label' in file.lower() or '.txt' in file.lower():
                print(f"\nFound potential label file: {file}")
                with open(os.path.join(assets_path, file), 'r') as f:
                    print(f.read())


def load_and_inspect_model(model_path):
    """Load model and inspect its structure"""
    print("\n2. Loading model and inspecting structure:")
    model = tf.saved_model.load(model_path)

    print("\nModel attributes:")
    print(dir(model))

    print("\nSignatures:")
    print(model.signatures.keys())

    
    for attr in dir(model):
        if 'label' in attr.lower() or 'class' in attr.lower():
            print(f"\nFound potential label-related attribute: {attr}")
            try:
                value = getattr(model, attr)
                print(value)
            except:
                print("Unable to read attribute")


def try_metadata_files(model_path):
    """Look for metadata files"""
    print("\n3. Looking for metadata files:")
    common_metadata_files = [
        'labels.txt',
        'class_names.txt',
        'metadata.json',
        'saved_model.pb',
        'keras_metadata.pb'
    ]

    for filename in os.listdir(model_path):
        if filename in common_metadata_files or 'label' in filename.lower():
            file_path = os.path.join(model_path, filename)
            print(f"\nFound: {filename}")
            try:
                if filename.endswith('.json'):
                    with open(file_path, 'r') as f:
                        print(json.load(f))
                elif filename.endswith('.txt'):
                    with open(file_path, 'r') as f:
                        print(f.read())
                else:
                    print(f"Binary file: {filename}")
            except Exception as e:
                print(f"Error reading file: {str(e)}")


def make_test_prediction(model_path):
    """Make a test prediction and analyze output"""
    print("\n4. Making test prediction to analyze output shape:")
    model = tf.saved_model.load(model_path)

    # Create a dummy input (adjust size according to your model's requirements)
    dummy_input = tf.zeros([1, 224, 224, 3])

    
    try:
        result = model(dummy_input, training=False)
        print("\nPrediction output shape:", result.shape)
        print("Number of classes:", result.shape[-1])

        # If it's a probability distribution
        if len(result.shape) == 2:
            print("\nThis suggests the model classifies into", result.shape[-1], "categories")
    except Exception as e:
        print(f"Error making prediction: {str(e)}")


def main():
    model_path = '/Users/todd.zhang/Downloads/dev/ai/monash/appliedpractice'

    print("=== Starting Model Inspection ===")
    print("model_path: ", model_path)

    
    inspect_saved_model(model_path)
    load_and_inspect_model(model_path)
    try_metadata_files(model_path)
    make_test_prediction(model_path)

    print("\n=== Inspection Complete ===")


if __name__ == "__main__":
    main()