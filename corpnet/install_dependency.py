import subprocess
import sys
import os


def install_tflite_dependencies():
    """Install TFLite Model Maker and required dependencies"""
    print("📦 Installing TFLite Model Maker and dependencies...")

    # List of required packages with versions
    requirements = [
        "tflite-model-maker",
        "tflite-support",
        "tensorflow==2.15.0",  # Specific version for compatibility
        "numpy",
        "pillow",
        "tensorflow-hub",
        "tensorflow-datasets"
    ]

    # Additional requirements that might be needed
    extra_requirements = [
        "tensorflow-metal",  # For Mac M1/M2 users
        "protobuf==3.20.3"  # Specific version to avoid conflicts
    ]

    def install_package(package):
        try:
            print(f"\nInstalling {package}...")
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                package,
                "--upgrade"
            ])
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing {package}: {str(e)}")
            return False

    # Install main requirements
    for req in requirements:
        if not install_package(req):
            print(f"Failed to install {req}")
            return False

    # Try installing extra requirements
    print("\n📦 Installing additional dependencies...")
    for req in extra_requirements:
        try:
            install_package(req)
        except:
            print(f"Note: Optional dependency {req} not installed")

    print("\n✅ Main dependencies installed successfully!")
    return True


def verify_installation():
    """Verify that all required modules are installed correctly"""
    print("\n🔍 Verifying installations...")

    required_modules = {
        "tensorflow": "import tensorflow as tf",
        "tflite_model_maker": "import tflite_model_maker as mm",
        "tflite_support": "import tflite_support",
        "tensorflow_hub": "import tensorflow_hub as hub",
        "PIL": "from PIL import Image",
        "numpy": "import numpy as np"
    }

    for module, import_statement in required_modules.items():
        try:
            exec(import_statement)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ Error importing {module}: {str(e)}")
            return False

    return True


def show_usage_example():
    """Show example code for using TFLite Model Maker"""
    print("""
🚀 Example usage:

import tensorflow as tf
import tflite_model_maker as mm
from tflite_model_maker import image_classifier

# Load data
data = image_classifier.DataLoader.from_folder('path/to/images')

# Create model
model = image_classifier.create(data)

# Evaluate the model
model.evaluate()

# Export the model
model.export('path/to/save/model')
""")


def main():
    print("🚀 Starting TFLite Model Maker installation process...")

    if install_tflite_dependencies():
        if verify_installation():
            print("\n🎉 Installation completed successfully!")
            show_usage_example()
        else:
            print("\n⚠️ Some modules couldn't be imported.")
            print("Try running the installation in a new terminal or virtual environment.")
    else:
        print("\n❌ Error during installation process.")
        print("Please check the error messages above and try again.")


if __name__ == "__main__":
    main()


# import subprocess
# import sys
# import os
#
#
# def install_dependencies():
#     """Install required dependencies"""
#     print("📦 Installing required dependencies...")
#
#     # List of required packages
#     requirements = [
#         "tensorflow-examples @ git+https://github.com/tensorflow/examples.git",
#         "tensorflow",
#         "tensorflow_hub",
#         "pillow",
#         "numpy"
#     ]
#
#     # Install each requirement
#     for requirement in requirements:
#         print(f"\nInstalling {requirement}...")
#         try:
#             subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
#         except subprocess.CalledProcessError as e:
#             print(f"❌ Error installing {requirement}: {str(e)}")
#             return False
#
#     print("\n✅ All dependencies installed successfully!")
#     return True
#
#
# def verify_installation():
#     """Verify that dependencies are installed correctly"""
#     print("\n🔍 Verifying installations...")
#
#     try:
#         # Try importing required modules
#         import tensorflow_examples
#         import tensorflow as tf
#         import tensorflow_hub as hub
#         from PIL import Image
#         import numpy as np
#
#         print("✅ All modules imported successfully!")
#         print("\nInstalled versions:")
#         print(f"TensorFlow: {tf.__version__}")
#         print(f"TensorFlow Hub: {hub.__version__}")
#         print(f"Pillow: {Image.__version__}")
#         print(f"NumPy: {np.__version__}")
#         return True
#
#     except ImportError as e:
#         print(f"❌ Import error: {str(e)}")
#         return False
#
#
# def main():
#     print("🚀 Starting dependency installation process...")
#
#     if install_dependencies():
#         if verify_installation():
#             print("\n🎉 Setup completed successfully!")
#             print("\nYou can now use the model with this example code:")
#             print("""
# import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow_examples.models.pix2pix import pix2pix
#
# # Load your model
# model = tf.saved_model.load('models/feature-vector-concat')
# """)
#         else:
#             print("\n⚠️ Installation verified but some modules couldn't be imported.")
#     else:
#         print("\n❌ Error during installation process.")
#
#
# if __name__ == "__main__":
#     main()