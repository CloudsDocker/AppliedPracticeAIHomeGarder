import kagglehub

# Download latest version
path = kagglehub.dataset_download("emmarex/plantdisease")
# saved On macOS: ~/Library/Caches/kagglehub/
# path = kagglehub.dataset_download("emmarex/plantdisease", path="./data", force_download=True)
print("Path to dataset files:", path)