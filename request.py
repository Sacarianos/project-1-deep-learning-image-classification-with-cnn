from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import json
import io
from google.auth import default
from google.auth.transport.requests import Request
import base64

# Load and preprocess the image
def preprocess_image(img_path, target_size=(32, 32)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1] range
    return img_array

# Convert image to base64
def image_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Path to the image you want to predict
img_path = 'images/360_F_177742846_umwpEr5OqwEQd4a9VyS7BGJX3tINNDe7.jpg'  # Replace with the path to your image

# Preprocess the image
img_array = preprocess_image(img_path)

# Prepare the request
input_data = np.expand_dims(img_array, axis=0).tolist()

# Send the request to TensorFlow Serving
endpoint_url = 'https://us-east1-aiplatform.googleapis.com/v1/projects/10609508497/locations/us-east1/endpoints/2868357556030406656:predict'

# Authenticate using service account credentials
credentials, project = default()
credentials.refresh(Request())

# Refresh the credentials (required if expired)
if credentials.expired:
    credentials.refresh(Request())

# Obtain the access token
access_token = credentials.token

# Add the authorization header to the request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {access_token}"
}

# Convert the image to base64 for the request (if needed)
base64_image = image_to_base64(img_path)

# Data structure for the request (with the base64 image)
data = json.dumps({
    "instances": [{"image": {"b64": base64_image}}]
})

# Make the prediction request
response = requests.post(endpoint_url, data=data, headers=headers)

# Print the predicted class
predictions = response.json()
print(predictions)

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Get the prediction values
prediction_values = predictions['predictions'][0]

# Get the index of the class with the highest probability
predicted_class_index = np.argmax(prediction_values)

# Get the class name corresponding to that index
predicted_class_name = class_names[predicted_class_index]

# Print the predicted class name
print(f"Predicted Class: {predicted_class_name}")
