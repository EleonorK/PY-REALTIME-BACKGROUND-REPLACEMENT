# Real-Time Video Background Replacement

This project is an implementation of real-time video background replacement using MediaPipe's machine learning segmentation and OpenCV.

## Description
The script captures video from a webcam, segments the person from the background using MediaPipe's Selfie Segmentation, and replaces the background with a chosen image or GIF. The application can be used for live streaming, video calls, or any scenario where you need dynamic background replacement.

## Techniques Used
- **MediaPipe Selfie Segmentation**: For segmenting the person from the background in real-time.
- **OpenCV**: For capturing video frames from the webcam and processing the video stream.
- **NumPy**: For numerical processing and manipulation of frames.
- **Pillow (PIL)**: For handling image operations when not working with GIFs.
- **ImageIO**: For reading GIF files and processing them as background frames.

## Steps
1. **Initialize MediaPipe Segmentation**:
   Set up the selfie segmentation model with `mp.solutions.selfie_segmentation.SelfieSegmentation`.

2. **Background Loading**:
   Load the desired background (image or GIF) and resize it to match the frame size of the video capture.

3. **Webcam Capture**:
   Start capturing video frames from the webcam using OpenCV's `VideoCapture`.

4. **Frame Processing**:
   For each frame from the webcam, convert it to RGB, and apply the segmentation model to get the mask.

5. **Background Replacement**:
   Use the mask to separate the person from the background and blend it with the chosen background image.

6. **Display Output**:
   Show the output in a window using OpenCV's `imshow` method.

7. **Exit on Key Press**:
   Allow the user to exit the loop and end the program by pressing the 'Esc' key.

