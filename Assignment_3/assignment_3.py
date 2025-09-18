import cv2
import numpy as np

# -------- 1. Sobel Edge Detection --------
def sobel_edge_detection(image_path, save_path="sobel_edges.jpg"):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)         # reduce noise
    sobel = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=1)  # dx=1, dy=1
    sobel = cv2.convertScaleAbs(sobel)
    cv2.imwrite(save_path, sobel)
    print(f"[INFO] Sobel edge detection saved as {save_path}")


# -------- 2. Canny Edge Detection --------
def canny_edge_detection(image_path, threshold_1=50, threshold_2=50, save_path="canny_edges.jpg"):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, threshold_1, threshold_2)  # edge detection
    cv2.imwrite(save_path, canny)
    print(f"[INFO] Canny edge detection saved as {save_path}")


# -------- 3. Template Matching --------
def template_match(image_path, template_path, save_path="template_match.jpg"):
    image = cv2.imread(image_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]

    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= 0.9)  # threshold = 0.9

    # Draw red rectangles on all matches
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite(save_path, image)
    print(f"[INFO] Template matching saved as {save_path}")


# -------- 4. Resizing (Image Pyramids) --------
def resize(image_path, scale_factor=2, up_or_down="up", save_path="resized.jpg"):
    image = cv2.imread(image_path)

    if up_or_down == "up":
        for _ in range(scale_factor):
            image = cv2.pyrUp(image)   # zoom in
    elif up_or_down == "down":
        for _ in range(scale_factor):
            image = cv2.pyrDown(image) # zoom out
    else:
        raise ValueError("up_or_down must be either 'up' or 'down'")

    cv2.imwrite(save_path, image)
    print(f"[INFO] Resized image saved as {save_path}")


if __name__ == "__main__":
    lambo_img = r"C:\Users\Sanjana\Downloads\lambo.png"
    shapes_img = r"C:\Users\Sanjana\Downloads\shapes-1.png"
    shapes_template = r"C:\Users\Sanjana\Downloads\shapes_template.jpg"

    sobel_edge_detection(lambo_img, save_path="sobel_edges.jpg")
    canny_edge_detection(lambo_img, 50, 50, save_path="canny_edges.jpg")
    template_match(shapes_img, shapes_template, save_path="template_match.jpg")
    resize(lambo_img, scale_factor=2, up_or_down="up", save_path="resized.jpg")