import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Define the class names for CropNet
class_names = {
    0: "Tomato - Healthy",
    1: "Tomato - Early Blight",
    2: "Tomato - Late Blight",
    3: "Tomato - Leaf Mold",
    4: "Tomato - Septoria Leaf Spot",
    5: "Tomato - Spider Mites",
    6: "Tomato - Target Spot",
    7: "Tomato - Yellow Leaf Curl Virus",
    8: "Tomato - Mosaic Virus",
    9: "Tomato - Bacterial Spot"
}


def preprocess_image(img_path):
    """Preprocess image for model input"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0,1]
    return img_array


def get_prediction_with_confidence(predictions):
    """Get prediction and confidence score"""
    # Convert predictions to probabilities if they aren't already
    if isinstance(predictions, tf.Tensor):
        predictions = predictions.numpy()

    # Get the predicted class index and confidence
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100

    # Get the human-readable label
    predicted_label = class_names.get(predicted_index, f"Unknown Class {predicted_index}")

    return {
        'label': predicted_label,
        'confidence': confidence,
        'index': predicted_index,
        'top_3': get_top_3_predictions(predictions[0])
    }


def get_top_3_predictions(predictions):
    """Get top 3 predictions with their confidences"""
    # Get indices of top 3 predictions
    top_indices = np.argsort(predictions)[-3:][::-1]

    return [
        {
            'label': class_names.get(idx, f"Unknown Class {idx}"),
            'confidence': float(predictions[idx] * 100)
        }
        for idx in top_indices
    ]


def main():
    # Path to your image
    img_path = '/Users/todd.zhang/Downloads/monash_tomato.jpeg'
    img_path = '/Users/todd.zhang/Downloads/pests-and-diseases-roses-520-500.jpeg'

    # Preprocess the image
    preprocessed_img = preprocess_image(img_path)

    # Load the model
    model = tf.saved_model.load('/Users/todd.zhang/Downloads/dev/ai/monash/appliedpractice')

    # Make prediction
    predictions = model(preprocessed_img, training=False)

    # Get prediction results
    result = get_prediction_with_confidence(predictions)

    # Print results
    print("\n🌿 Plant Disease Detection Results:")
    print(f"\nPrimary Prediction:")
    print(f"📍 Condition: {result['label']}")
    print(f"📊 Confidence: {result['confidence']:.2f}%")

    print("\nTop 3 Possibilities:")
    for i, pred in enumerate(result['top_3'], 1):
        print(f"{i}. {pred['label']} ({pred['confidence']:.2f}%)")

    # If it's a disease, provide basic care instructions
    if "Healthy" not in result['label']:
        print("\n⚕️ Care Recommendations:")
        care_instructions = get_care_instructions(result['index'])
        print(care_instructions)


def get_care_instructions(disease_index):
    """Get care instructions based on the disease"""
    care_instructions = {
        1: """Early Blight Treatment:
- Remove affected leaves
- Improve air circulation
- Apply fungicide if severe
- Water at the base of plant""",
        2: """Late Blight Treatment:
- Remove infected plants
- Apply copper-based fungicide
- Improve drainage
- Space plants properly""",
        3: """Leaf Mold Treatment:
- Reduce humidity
- Improve air circulation
- Remove affected leaves
- Apply fungicide if needed""",
        # Add more care instructions for other diseases
    }

    return care_instructions.get(disease_index,
                                 "Please consult with a plant specialist for specific treatment recommendations.")


if __name__ == "__main__":
    main()