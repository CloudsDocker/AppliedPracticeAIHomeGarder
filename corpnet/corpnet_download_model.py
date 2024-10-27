import kagglehub
import os
import shutil
from pathlib import Path


def download_and_save_model(model_name="google/cropnet/tensorFlow2/feature-vector-concat"):
    """
    Download model from Kaggle and save to project's models directory

    Args:
        model_name: Kaggle model identifier
    """
    try:
        # Get current project directory (where script is located)
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Create models directory in project folder
        models_dir = os.path.join(project_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)

        print(f"📂 Project directory: {project_dir}")
        print(f"📂 Models directory: {models_dir}")
        print(f"📥 Downloading model: {model_name}")

        # Download the model
        downloaded_path = kagglehub.model_download(model_name)
        print(f"✅ Downloaded to temporary path: {downloaded_path}")

        # Create specific model directory in project's models folder
        model_save_dir = os.path.join(models_dir, Path(model_name).name)
        os.makedirs(model_save_dir, exist_ok=True)

        print(f"📁 Copying to project directory: {model_save_dir}")

        # Copy all files from downloaded path to project's models directory
        if os.path.isdir(downloaded_path):
            # If downloaded_path is a directory, copy its contents
            for item in os.listdir(downloaded_path):
                source = os.path.join(downloaded_path, item)
                destination = os.path.join(model_save_dir, item)
                if os.path.isdir(source):
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)
        else:
            # If downloaded_path is a file, copy it directly
            shutil.copy2(downloaded_path, model_save_dir)

        print(f"✨ Model saved successfully to: {model_save_dir}")

        # List contents of saved model directory
        print("\n📄 Saved model contents:")
        for item in os.listdir(model_save_dir):
            print(f"  - {item}")

        return model_save_dir

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise


def verify_model_files(model_path):
    """Verify that all necessary model files are present"""
    required_files = ['saved_model.pb']
    required_dirs = ['variables']

    all_present = True

    print("\n🔍 Verifying model files:")

    # Check required files
    for file in required_files:
        file_path = os.path.join(model_path, file)
        exists = os.path.exists(file_path)
        print(f"  - {file}: {'✅' if exists else '❌'}")
        all_present &= exists

    # Check required directories
    for dir_name in required_dirs:
        dir_path = os.path.join(model_path, dir_name)
        exists = os.path.exists(dir_path)
        print(f"  - {dir_name}/: {'✅' if exists else '❌'}")
        all_present &= exists

        if exists and dir_name == 'variables':
            # Check for variables files
            var_files = ['variables.data-00000-of-00001', 'variables.index']
            for var_file in var_files:
                var_path = os.path.join(dir_path, var_file)
                var_exists = os.path.exists(var_path)
                print(f"    - {var_file}: {'✅' if var_exists else '❌'}")
                all_present &= var_exists

    return all_present


def main():
    try:
        # Download and save the model
        model_path = download_and_save_model()

        # Verify the saved model
        if verify_model_files(model_path):
            print("\n✅ Model successfully saved and verified")
            print(f"📍 Model path: {model_path}")

            # Print example usage
            print("\n📝 Example usage:")
            print("import tensorflow as tf")
            print(f"model = tf.saved_model.load('{model_path}')")
        else:
            print("\n⚠️ Some model files are missing")

    except Exception as e:
        print(f"\n❌ An error occurred during execution: {str(e)}")


if __name__ == "__main__":
    main()