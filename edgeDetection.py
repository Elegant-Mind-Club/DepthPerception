import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# Constants
IMG1_PATH = 'left.JPG'
IMG2_PATH = 'right.JPG'

IMG_CROP_START_X = 1250
IMG_CROP_START_Y = 1650
IMG_CROP_WIDTH = 800
IMG_CROP_HEIGHT = 450

# best practice is 1,3,5,7
KERNEL_SIZE = 3
TOLERANCE = 1e-5


def mse(patch1, patch2):
    """Compute the Mean Squared Error between two image patches."""
    err = np.sum((patch1.astype("float") - patch2.astype("float")) ** 2)
    err /= float(patch1.shape[0] * patch1.shape[1])
    return err


def initilizeImage(imgPath, imgCropStartX, imgCropStartY, imgCropWidth, imgCropHeight):
    # Return cropped image
    return cv2.imread(imgPath)[imgCropStartY:imgCropStartY+imgCropHeight, imgCropStartX:imgCropStartX+imgCropWidth]


def extractColors(img1, img2, color):

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

# Color
def visualCortexV1():
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


def convolute(img1, img2, kernel_size):
    # Depth
    return find_best_match(img1, img2, kernel_size)


def find_best_match(image1, image2, kernel_size):
    """Find best MSE match for each kernel position in image1 by sliding a kernel in image2."""
    height, width = image1.shape[:2]
    match_map = np.zeros((height - kernel_size + 1, width - kernel_size + 1))
    match_locations = []

    for y in range(0, height - kernel_size + 1, KERNEL_SIZE):

        for x1 in range(0, width - kernel_size + 1):
            # print(y, x1, "\n")
            patch1 = image1[y:y + kernel_size, x1:x1 + kernel_size]
            best_mse = float('inf')  # Initialize with a high value for MSE
            best_mse_x2 = 0 # x value (where kernel begins) of best match in img2

            # Only slide kernel if there is an edge in img1
            if np.mean(patch1) != 0:
                for x2 in range(x1, width - kernel_size + 1):
                    patch2 = image2[y:y + kernel_size, x2:x2 + kernel_size]
                    current_mse = mse(patch1, patch2)

                    # Update the best MSE value if the current one is better
                    if current_mse < best_mse:
                        best_mse = current_mse
                        best_mse_x2 = x2

                # for patch in img1, store the best MSE value
                match_map[y, x1] = best_mse

                # If MSE is zero (or very close to zero), store the location
                if np.isclose(best_mse, 0, atol=TOLERANCE):
                    match_locations.append((y, x1, best_mse_x2))

    # print position map
    np.set_printoptions(threshold=np.inf)
    print("First 20 Match locations: ", match_locations[:20])
    match_locations.sort()
    # print(match_locations)


    # Different way of generating position image, with color

    visualization = np.ones((height, width*2, 3), dtype=np.uint8) * 255

    visualization[:, :width, 0] = image1
    visualization[:, width:, 0] = image2

    for (y, x1, x2) in match_locations:

        # draw line in different colors
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.line(visualization, (x1, y), (x2+width, y), color, 1)

        # cv2.putText(visualization, f"{y}", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        # cv2.putText(visualization, f"{y}", (x2+width, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)




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

    cv2.imshow("Position image", position_image)
    # cv2.imshow("Visualization", visualization)
    # cv2.waitKey(0)

    return match_map

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
    
    # Initial Images
    imagesBefore = np.concatenate((img1, img2), axis=1)
    cv2.imshow("Initial images", imagesBefore)
    # cv2.waitKey(0)

    r1, r2 = extractColors(img1, img2, 'red')
    g1, g2 = extractColors(img1, img2, 'green')
    b1, b2 = extractColors(img1, img2, 'blue')

    # cv2.namedWindow(f"Extracted red", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Extracted red", np.concatenate((r1, r2), axis=1))
    # cv2.imshow(f"Extracted blue", np.concatenate((b1, b2), axis=1))


    # Standard
    edge1 = detectEdges(img1)
    edge2 = detectEdges(img2)
    # cv2.imshow('Canny Edge Detection', np.concatenate((edge1, edge2), axis=1))

    # Color
    edge1_r = detectEdges(r1)
    edge2_r = detectEdges(r2)
    # cv2.imshow('Canny Edge Detection w color', np.concatenate((edge1_r, edge2_r), axis=1))
    cv2.waitKey(0)

    blended_initial = superimpose(edge1, edge2, 0.3)
    cv2.imshow("Edges Superimposed", blended_initial)

    blended_r = superimpose(edge1_r, edge2_r, 0.3)
    cv2.imshow("Red Edges Superimposed", blended_r)


    # match_map = convolute(edge1, edge2, KERNEL_SIZE)
    # cv2.imshow('Match Map', match_map)
    # # # print(match_map)

    # # Adding color to the match_map
    # colored_match_map = addColor(match_map, (0, 0, 255))
    # cv2.imshow("Colored Match map", colored_match_map)


    # # blended_edge_match = superimpose(edge1, colored_match_map, 0.3)
    # # cv2.imshow("Img with match Superimposed", blended_edge_match)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
