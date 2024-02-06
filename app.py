import numpy as np
import mediapipe as mp
import cv2
from PIL import Image, ImageFilter
import imageio
import os

# Initialize the selfie segmentation model
segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

# Function to check if the file is a GIF
def is_gif(file_path):
    _, ext = os.path.splitext(file_path)
    return ext.lower() == ".gif"

# Function to load the background image or GIF
def load_background(file_path, frame_size):
    if is_gif(file_path):
        return imageio.mimread(file_path)
    else:
        image = Image.open(file_path)
        image = image.resize(frame_size, Image.ANTIALIAS)
        return [np.array(image.convert("RGB"))]

# Load the background
background_path = "back/op7.gif"  # Replace with the path to your background file
frame_size = (1100, 800)  # Set the desired frame size
background_frames = load_background(background_path, frame_size)

# Open the camera capture
cap = cv2.VideoCapture(0)

# Set the capture properties to the desired frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

# Create a named window for displaying the output
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("output", frame_size[0], frame_size[1])

# Start the video capture loop
frame_index = 0
while cap.isOpened():
    # Read a frame from the capture
    ret, frame = cap.read()

    # Get the dimensions of the frame
    frame_height, frame_width, _ = frame.shape

    # Convert the frame from BGR to RGB
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the selfie segmentation model
    results = segmentation.process(RGB)
    mask = results.segmentation_mask

    # Stack the mask channels to match the frame dimensions
    rsm = np.stack((mask,) * 3, axis=-1)

    # Create a condition based on the mask threshold
    condition = rsm > 0.6

    # Reshape the condition to match the frame dimensions
    condition = np.reshape(condition, (frame_height, frame_width, 3))

    # Get the next frame from the background frames
    background_frame = background_frames[frame_index % len(background_frames)]
    background_frame = cv2.cvtColor(background_frame, cv2.COLOR_RGBA2BGR)  # Convert to BGR
    background_frame = cv2.resize(background_frame, (frame_width, frame_height))

   # Convert the condition to BGR
    condition_uint8 = (condition * 255).astype(np.uint8)
    condition_bgr = cv2.cvtColor(condition_uint8, cv2.COLOR_RGB2BGR)

    # Apply the condition to the frame and background
    output = np.where(condition_bgr, frame, background_frame)

    # Display the output
    cv2.imshow("output", output)

    # Check for the 'Esc' key press to exit the loop
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

    # Increment the frame index to get the next frame from the background frames
    frame_index += 1

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
