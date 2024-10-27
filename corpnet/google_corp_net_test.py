import tensorflow as tf

print('====loading model =======')
# Load the model from the directory where 'saved_model.pb' is located
# model = tf.keras.models.load_model('/Users/todd.zhang/Downloads/dev/ai/monash/appliedpractice')
print(("------ loaded model ----"))
from tensorflow.keras.preprocessing import image
import numpy as np



def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Change target size if needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Scale to [0, 1] range
    return img_array

img_path = '/Users/todd.zhang/Downloads/Monstera-Problems.jpg'
img_path = '/Users/todd.zhang/Downloads/monash_tomato.jpeg'
preprocessed_img = preprocess_image(img_path)
# print(preprocessed_img.shape)
# predictions = model.predict(preprocessed_img)
# predicted_class = np.argmax(predictions, axis=1)
# print(f"Predicted class: {predicted_class}")

# Load the model as a SavedModel
model = tf.saved_model.load('/Users/todd.zhang/Downloads/dev/ai/monash/appliedpractice')
# Explore the model to understand its structure and available attributes
print("Model attributes:", dir(model))
# Load the model
model = tf.saved_model.load('/Users/todd.zhang/Downloads/dev/ai/monash/appliedpractice')

# Assuming `preprocessed_img` is already prepared as an input tensor
predictions = model(preprocessed_img, training=False)  # Set training to False for inference
# print(predictions)
print("... predictions:", predictions)
# print(predictions)
# Assuming model outputs are in 'predictions'
class_names = ["Label 1", "Label 2", "Label 3", "Label 4", "Label 5", "Label 6"]  # replace with actual class names

# Get the index of the highest probability
predicted_index = tf.argmax(predictions, axis=1).numpy()[0]
predicted_label = class_names[predicted_index]

print("Predicted label:", predicted_label)



#
# # Prepare your input data as a Tensor
# preprocessed_img = tf.convert_to_tensor(preprocessed_img)
#
# # Run inference (this may vary based on the model's output signatures)
# # Check the model's available signatures
# print("Available signatures:", list(model.signatures.keys()))
#
# # Use the 'serving_default' signature (if it exists) for prediction
# inference = model.signatures["serving_default"]
# predictions = inference(preprocessed_img)
# print("Predictions:", predictions)