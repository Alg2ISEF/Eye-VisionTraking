import cv2
import numpy as np
from PIL import Image

# Define grid and marker parameters
squares_x = 5
squares_y = 7
square_length = 40  
marker_length = 30

# Initialize the ChArUco dictionary and board layout
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)

# Generate a high-resolution image layout (e.g., A4 dimensions at 300 DPI: 2480 x 3508 pixels)
img_size = (2480, 3508)
board_img = board.generateImage(img_size)

# Save as a clean vector-scaled PDF using Pillow
pil_img = Image.fromarray(board_img)
pil_img.save("charuco_board.pdf", "PDF", resolution=300.0)

print("ChArUco board successfully generated and saved as charuco_board.pdf")