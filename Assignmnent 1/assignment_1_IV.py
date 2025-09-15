import cv2

def print_image_information(image):
    img = cv2.imread(image)

    height, width, channels = img.shape
    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
    print("Size (total number of values):", img.size)
    print("Data type:", img.dtype)

print_image_information(r"C:\Users\Sanjana\Downloads\lena-1.png")
