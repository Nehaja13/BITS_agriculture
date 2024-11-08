# utils.py

# Dictionary mapping disease names to details
disease_dic = {
    "Powdery Mildew": "A fungal disease affecting leaves, causing white powdery spots.",
    "Leaf Spot": "Characterized by spots on leaves, usually caused by bacteria or fungi.",
    # Add other diseases and their descriptions as needed
}

# (Optional) Helper function for formatting or data conversion
def preprocess_image(image):
    # Code to preprocess images for model input
    return processed_image

# (Optional) Helper function to fetch data based on input parameters
def get_disease_info(disease_name):
    return disease_dic.get(disease_name, "Disease not found.")
