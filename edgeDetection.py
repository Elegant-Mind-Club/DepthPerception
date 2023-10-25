import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Constants
IMG1_PATH = 'IMG_7210.JPG'

IMG2_PATH = 'IMG_7212.JPG'

# IMG_CROP_START_X = 1300
# IMG_CROP_START_Y = 1500
# IMG_CROP_WIDTH = 1200
# IMG_CROP_HEIGHT = 600

# IMG_CROP_START_X = 1300
# IMG_CROP_START_Y = 1700
# IMG_CROP_WIDTH = 300
# IMG_CROP_HEIGHT = 200

IMG_CROP_START_X = 0
IMG_CROP_START_Y = 0
IMG_CROP_WIDTH = 10000
IMG_CROP_HEIGHT = 10000

KERNEL_SIZE = 7
TOLERANCE = 1e-5
MAXIMUM_DISPLACEMENT = cv2.imread(IMG1_PATH).shape[1] / 15

def mse(patch1, patch2):
    """Compute the Mean Squared Error between two image patches."""
    err = np.sum((patch1.astype("float") - patch2.astype("float")) ** 2)
    err /= float(patch1.shape[0] * patch1.shape[1])
    return err


def initilizeImage(imgPath, imgCropStartX, imgCropStartY, imgCropWidth, imgCropHeight):
    # Return cropped image
    img = cv2.imread(imgPath)[imgCropStartY:imgCropStartY+imgCropHeight, imgCropStartX:imgCropStartX+imgCropWidth]
    return img

# def extractColors(img1, img2, color):

    #convert the BGR image to HSV colour space
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # Set the lower and upper bounds for the color hues - need to find this online
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_green = np.array([50,100,50])
    upper_green = np.array([70,255,255])
    lower_blue = np.array([60, 35, 140]) 
    upper_blue = np.array([180, 255, 255]) 

    match color:
        case 'red':
            lower_color = lower_red
            upper_color = upper_red
        case 'green':
            lower_color = lower_green
            upper_color = upper_green
        case 'blue':
            lower_color = lower_blue
            upper_color = upper_blue
        case 'white':
            lower_color = lower_white
            upper_color = upper_white
        case 'black':
            lower_color = lower_black
            upper_color = upper_white
        case _:
            print("Error in color")


    # Create a mask for the selected colour using inRange function
    mask1 = cv2.inRange(hsv1, lower_color, upper_color)
    mask2 = cv2.inRange(hsv2, lower_color, upper_color)

    # Perform bitwise and on the original image arrays using the mask
    result1 = cv2.bitwise_and(img1, img1,mask=mask1)
    result2 = cv2.bitwise_and(img2, img2,mask=mask2)

    return result1, result2


def visualCortexV1():
    # Color
    pass

# Curves and lines
def detectEdges(img):
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    # Canny Edge Detection. 
    # 5 step process: image smoothening -> finding intensity gradients -> 
    # non-max suppression -> double threshold -> hysteresis edge tracking
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    return edges

def visualCortexV2V3(img):
    # Curves and lines
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    return edges


def convolute(img1, img2, kernel_size, originalImage):
    # Depth
    return find_best_match(img1, img2, kernel_size, originalImage)

# alpha: transparency value for blending for img 1


def superimpose(img1, img2, alpha):
    # make sure both are of the same datatype
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)

    assert img1.dtype == img2.dtype
    print(img1.dtype)
    print(img2.dtype)

    # shape[1]: width, shape[0]: height
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    blended = cv2.addWeighted(img1, alpha, img2_resized, 1-alpha, 0)

    return blended


def find_best_match(image1Edges, image2Edges, kernel_size, image1):
    """Find best MSE match for each kernel position in image1 by sliding a kernel in image2."""
    height, width = image1Edges.shape
    match_map = np.zeros((height - kernel_size + 1, width - kernel_size + 1))
    match_locations = []

    for y in range(0, height - kernel_size + 1, 3):
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
                if np.isclose(best_mse, 0, atol=1e-10) and best_mse_x2 - x1 < MAXIMUM_DISPLACEMENT:
                    match_locations.append((y, x1, best_mse_x2))

    # print position map
    np.set_printoptions(threshold=np.inf)
    match_locations.sort()
    # print(match_locations)

    # Generate position image. Initialize an all-black image
    position_image = np.ones((height, width), dtype=np.uint8) * 255
    # prev = -1
    for (y, x1, x2) in match_locations:
        # if y == prev:
        #     continue
        # prev = y
        position_image[y, x2] = 0  # Set (y, x2) to white
        position_image[y, x1] = 0  # Set (y, x1) to white

        # Draw a line between (y, x1) and (y, x2)
        cv2.line(position_image, (x1, y), (x2, y), (0, 0, 255), 1)
        cv2.line(image1, (x1, y), (x2, y), (0, 0, 255), 1)

    cv2.imshow("Position image", position_image)
    cv2.imshow("Image with lines", image1)
    cv2.waitKey(0)

    return match_map

def addColor(img, color):
    # need to convert it to float32
    img = np.float32(img)

    # creating a mask for white regions (simpler than color extraction)
    _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # convert image to BGR
    colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    colored[mask>0] = color

    return colored

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

    blended_initial = superimpose(img1, img2, 0.3)
    cv2.imshow("Superimposed images", blended_initial)
    cv2.waitKey(0)

    match_map = convolute(img1Edges, img2Edges, KERNEL_SIZE, blended_initial)
    cv2.imshow('Match Map', match_map)
    # print(match_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
