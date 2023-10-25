import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# Constants
IMG1_PATH = '/Users/alex/Desktop/UCLA school work/Research/Elegant mind/Depth perception/images/20cm copy.jpg'

IMG2_PATH = '/Users/alex/Desktop/UCLA school work/Research/Elegant mind/Depth perception/images/10cm copy.jpg'

print(124)

# IMG_CROP_START_X = 1250
# IMG_CROP_START_Y = 1650
# IMG_CROP_WIDTH = 800
# IMG_CROP_HEIGHT = 450

# IMG_CROP_START_X = 1300
# IMG_CROP_START_Y = 1700
# IMG_CROP_WIDTH = 300
# IMG_CROP_HEIGHT = 200

IMG_CROP_START_X = 0
IMG_CROP_START_Y = 0
IMG_CROP_WIDTH = 10000
IMG_CROP_HEIGHT = 10000

KERNEL_SIZE = 5


def mse(patch1, patch2):
    """Compute the Mean Squared Error between two image patches."""
    err = np.sum((patch1.astype("float") - patch2.astype("float")) ** 2)
    err /= float(patch1.shape[0] * patch1.shape[1])
    return err


def initilizeImage(imgPath, imgCropStartX, imgCropStartY, imgCropWidth, imgCropHeight):
    # Return cropped image

    return cv2.imread(imgPath)[imgCropStartY:imgCropStartY+imgCropHeight, imgCropStartX:imgCropStartX+imgCropWidth]


def visualCortexV1():
    # Color
    pass


def visualCortexV2V3(img):
    # Curves and lines
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    return edges


def visualCortexV4(img1Edges, img2Edges, img1):
    # Depth
    return find_best_match(img1Edges, img2Edges, KERNEL_SIZE, img1)


def find_best_match(image1Edges, image2Edges, kernel_size, image1):
    """Find best MSE match for each kernel position in image1 by sliding a kernel in image2."""
    height, width = image1Edges.shape
    match_map = np.zeros((height - kernel_size + 1, width - kernel_size + 1))
    match_locations = []

    for y in range(0, height - kernel_size + 1, KERNEL_SIZE):
        for x1 in range(0, width - kernel_size + 1):
            # print(y, x1, "\n")
            patch1 = image1Edges[y:y + kernel_size, x1:x1 + kernel_size]
            best_mse = float('inf')  # Initialize with a high value for MSE
            best_mse_x2 = 0

            # Only slide kernel if there is an edge
            if np.mean(patch1) != 0:
                for x2 in range(x1, width - kernel_size + 1):
                    patch2 = image2Edges[y:y +
                                         kernel_size, x2:x2 + kernel_size]
                    current_mse = mse(patch1, patch2)

                    # Update the best MSE value if the current one is better
                    if current_mse < best_mse:
                        best_mse = current_mse
                        best_mse_x2 = x2

                # Store the best MSE value for this position of the kernel in image1
                match_map[y, x1] = best_mse

                # If MSE is zero (or very close to zero), store the location
                if np.isclose(best_mse, 0, atol=1e-10):
                    match_locations.append((y, x1, best_mse_x2))

    # print position map
    np.set_printoptions(threshold=np.inf)
    match_locations.sort()
    # print(match_locations)

    # Generate position image. Initialize an all-black image
    position_image = np.ones((height, width), dtype=np.uint8) * 255
    prev = -1
    for (y, x1, x2) in match_locations:
        if y == prev:
            continue
        prev = y
        position_image[y, x2] = 0  # Set (y, x2) to white
        position_image[y, x1] = 0  # Set (y, x1) to white

        # Draw a line between (y, x1) and (y, x2)
        cv2.line(position_image, (x1, y), (x2, y), (0, 0, 255), 1)
        cv2.line(image1, (x1, y), (x2, y), (0, 0, 255), 1)

    cv2.imshow("Position image", position_image)
    cv2.imshow("Image with lines", image1)
    cv2.waitKey(0)

    return match_map


def main():
    img1 = initilizeImage(IMG1_PATH, IMG_CROP_START_X,
                          IMG_CROP_START_Y, IMG_CROP_WIDTH, IMG_CROP_HEIGHT)
    img2 = initilizeImage(IMG2_PATH, IMG_CROP_START_X,
                          IMG_CROP_START_Y, IMG_CROP_WIDTH, IMG_CROP_HEIGHT)

    # Images before
    imagesBefore = np.concatenate((img1, img2), axis=1)
    cv2.namedWindow('Resized Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Initial images", imagesBefore)
    cv2.waitKey(0)

    # Brain process
    visualCortexV1()
    img1Edges = visualCortexV2V3(img1)
    img2Edges = visualCortexV2V3(img2)
    imagesEdges = np.concatenate((img1Edges, img2Edges), axis=1)
    cv2.namedWindow('Resized Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Canny Edge Detection', imagesEdges)
    cv2.waitKey(0)

    match_map = visualCortexV4(img1Edges, img2Edges, img1)
    cv2.imshow('Match Map', match_map)
    # print(match_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
