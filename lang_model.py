# import torch
# from PIL import Image
# import sys
# from langchain_community.llms import CTransformers

# # Load YOLOv5 model
# model = torch.hub.load('C:/Users/VANSHIKA/.cache/torch/hub/ultralytics_yolov5_master', 'custom', path='best.pt', source='local' ,force_reload=True) 

# # Function to perform object detection using YOLOv5
# def perform_object_detection(image_path):
#     img = Image.open(image_path).convert("RGB")
#     results = model(img)
#     return results

# # Function to extract diseases from YOLOv5 results
# # Modified function to extract only the disease with the highest probability
# def extract_diseases(yolov5_results):
#     # Check if there are any detections
#     if len(yolov5_results.xyxy[0]) > 0:
#         # Sort detections by confidence score (the 4th index of each detection is the confidence score)
#         sorted_detections = sorted(yolov5_results.xyxy[0], key=lambda x: x[4], reverse=True)
#         # Get the name of the disease with the highest confidence score
#         highest_confidence_disease = model.names[int(sorted_detections[0][5])]
#         print(highest_confidence_disease)
#         return [highest_confidence_disease]  # Return as a list for compatibility with the generate_response function
#     else:
#         # Return an empty list if no detections are made
#         return []
# # The rest of your script can remain the same


# # Function to load the LLaMA model
# def load_llm():
#     llm = CTransformers(
#         model="llama-2-7b-chat.Q5_K_S.gguf",
#         model_type="llama",
#         max_new_tokens=100000,  # Adjust this value as needed
#         temperature=0.5
#     )
#     return llm

# # Function to generate a response to a query using LLaMA
# # Modified generate_response function to accept an additional text query
# def generate_response(disease, text_query):
#     llm = load_llm()  # Load the LLaMA model
#     # Since there's only one disease, directly use it without joining
#     combined_query = f"{disease}. {text_query}" if disease else text_query
#     response = llm(combined_query)  # Generate a response to the combined query
#     return response


# # Example usage with an additional text query
# image_path = 'C:/yoloskin/images/psoriasis.jpeg' 
# yolov5_results = perform_object_detection(image_path)
# # Extract the disease with the highest probability (now returns only one disease)
# diseases = extract_diseases(yolov5_results)
# # Since we're now expecting a single disease, we can pass it directly without a list
# disease = diseases[0] if diseases else None
# text_query = input("Enter your text query: ")  # User-provided text query
# response = generate_response(disease, text_query)
# print("Response:", response)

import torch
from PIL import Image
import sys
import numpy as np
from langchain_community.llms import CTransformers

# Load YOLOv5 model


# Function to perform object detection using YOLOv5
def perform_object_detection(image_file):
    model = torch.hub.load("C:/Users/VANSHIKA/.cache/torch/hub/ultralytics_yolov5_master", 'custom', path='best.pt', source='local', force_reload=True)
    image =  Image.open(image_file)
    image = image.convert("RGB")
   # image_np = np.array(image)
    results = model(image)
    if len(results.xyxy[0]) > 0:
        sorted_detections = sorted(results.xyxy[0], key=lambda x: x[4], reverse=True)
        highest_confidence_disease = model.names[int(sorted_detections[0][5])]
        return highest_confidence_disease
    else:
        return "No disease detected"

# Function to load the LLaMA model
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.Q5_K_S.gguf",
        model_type="llama",
        max_new_tokens=150,
        temperature=0.5
    )
    return llm

# Function to generate a response to a query using LLaMA
def generate_response(context, text_query):
    llm = load_llm()
    print("finding answer")
    combined_query = f"{context} {text_query}"
    response = llm(combined_query)
    return response



# image_path = input('Enter the path to your image: ') 
# #input("Enter the path to your image: ")  # User provides the image path
# disease = perform_object_detection(image_path)

# context = f"The detected disease is {disease}."  # Initial context based on disease detection

# print(context) # Inform the user about the detected disease

# if __name__ == "__main__":
#     while True:
#         text_query = input("Ask me anything about the disease or type 'exit' to quit: ")
#         if text_query.lower() == 'exit':
#             print("Exiting chatbot. Goodbye!")
#             break
#         print('Working')
#         response = generate_response(context, text_query)
#         print("Response:", response)

