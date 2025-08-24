import cv2
import os

def save_camera_information():
    # Create output directory if it doesn't exist
    output_dir = os.path.expanduser("~/IKT213_lastname/assignment_1/solutions/")
    os.makedirs(output_dir, exist_ok=True)

    # File path
    file_path = os.path.join(output_dir, "camera_outputs.txt")

    # Open default camera (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get camera properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cap.release()

    with open(file_path, "w") as f:
        f.write(f"fps: {int(fps)}\n")
        f.write(f"height: {int(height)}\n")
        f.write(f"width: {int(width)}\n")

    print(f"Camera information saved to: {file_path}")

save_camera_information()
