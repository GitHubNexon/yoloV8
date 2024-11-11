import cv2
import sys
import os
from ultralytics import YOLO
import mobile  

# Load the trained model
model_path = r"C:\4th Year\Thesis-Projects\YoloV8\ultralytics\runs\CropV1Trained\cropV1\weights\best.pt"
model = YOLO(model_path)

# Define result folder path
results_path = r"C:\4th Year\Thesis-Projects\YoloV8\ultralytics\results"
os.makedirs(results_path, exist_ok=True)  # Ensure the folder exists

def get_incremental_filename(base_path, base_name, extension):
    """
    Generates a unique filename with an incrementing number in the results folder.
    For example, if 'webcam_result1.avi' exists, it will create 'webcam_result2.avi'.
    """
    i = 1
    while True:
        filename = f"{base_name}{i}{extension}"
        if not os.path.exists(os.path.join(base_path, filename)):
            return os.path.join(base_path, filename)
        i += 1

def detect_from_webcam():
    # Open the webcam
    cap = cv2.VideoCapture(0)  # Change to your camera index if necessary
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        sys.exit()

    # Get an incremented filename for the video output
    video_filename = get_incremental_filename(results_path, "webcam_result", ".avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 20.0  # Frames per second
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

    print(f"Recording video to {video_filename}. Press 'q' to stop recording.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from camera.")
            break

        # Perform detection
        results = model(frame)

        # Ensure results is a list and take the first item
        result = results[0]  # Get the first detection result

        # Render results on the frame
        frame = result.plot()  # Use plot on the first result

        # Display the frame with detection boxes
        cv2.imshow('Crop Detection - Webcam', frame)

        # Write the frame to the video file
        out.write(frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera, video writer, and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved at {video_filename}")


def detect_from_mobilecam():
    # URL for the mobile camera stream
    video_url = 'http://192.168.100.192:4747/video'
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        print("Error: Could not access the mobile camera.")
        sys.exit()

    video_filename = get_incremental_filename(results_path, "mobilecam_result", ".avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 40.0
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
    print(f"Recording video to {video_filename}. Press 'q' to stop recording.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from mobile camera.")
            break

        results = model(frame)
        frame = results[0].plot()  # Render results on the frame

        cv2.imshow('Crop Detection - MobileCam', frame)
        out.write(frame)

        # Add slight delay after showing the frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved at {video_filename}")


def detect_from_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        sys.exit()

    # Perform detection
    results = model(image)

    # Check if the results are a list and access the first result
    if isinstance(results, list):
        result = results[0]  # Get the first detection result
    else:
        result = results  # Otherwise, directly use the results object

    # Render results on the image
    image = result.plot()  # Now use plot() on the individual result

    # Save the image with detections to the results folder with an incremented filename
    image_filename = get_incremental_filename(results_path, "image_output", ".jpg")
    cv2.imwrite(image_filename, image)
    print(f"Image saved at {image_filename}")

    # Display the image with detection boxes
    cv2.imshow('Crop Detection - Image', image)

    # Wait until a key is pressed and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def detect_from_video(video_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit()

    # Get an incremented filename for the output video
    video_filename = get_incremental_filename(results_path, "video_output", ".avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi file format
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Initialize VideoWriter to save the processed video
    out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
    print(f"Saving processed video as {video_filename}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished processing video.")
            break

        # Perform detection
        results = model(frame)

        # Check if the results are a list and access the first result
        if isinstance(results, list):
            result = results[0]  # Get the first detection result
        else:
            result = results  # Otherwise, directly use the results object

        # Render results on the frame
        frame = result.plot()  # Now use plot() on the individual result

        # Display the frame with detection boxes
        cv2.imshow('Crop Detection - Video', frame)

        # Write the processed frame to the output video file
        out.write(frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video, video writer, and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved at {video_filename}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python crop_detect.py <mode> [file_path]")
        print("Modes: webcam, mobilecam, image, video")
        sys.exit()

    mode = sys.argv[1].lower()

    if mode == 'webcam':
        detect_from_webcam()
    elif mode == 'mobilecam':
        detect_from_mobilecam()
    elif mode == 'image' and len(sys.argv) == 3:
        image_path = sys.argv[2]
        detect_from_image(image_path)
    elif mode == 'video' and len(sys.argv) == 3:
        video_path = sys.argv[2]
        detect_from_video(video_path)
    else:
        print("Invalid usage. For image or video detection, provide a file path.")
        sys.exit()

if __name__ == "__main__":
    main()


# Scripts to start detection

# Webcam detection:
# python crop_detect.py webcam

# MobileCam detection:
# python crop_detect.py mobilecam


# Image detection:
# python crop_detect.py image path/to/your/image.jpg

# Video detection:
# python crop_detect.py video path/to/your/video.mp4


# some comment here
