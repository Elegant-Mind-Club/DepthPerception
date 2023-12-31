import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os

# Constants
IMG1_PATH = '10cmDownscaled.jpg'

IMG2_PATH = '20cmDownscaled.jpg'

VIDEO_PATH = 'video/testVideo.MOV'

IMAGES_FROM_VIDEO_PATH = 'imagesFromVideo'

IMAGE_SCALE_DOWN_FACTOR = 0.3

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
MAXIMUM_DISPLACEMENT = cv2.imread(IMG1_PATH).shape[1] / 3

def resize_image(image, scaleFactor):
    """
    Resize the given image by the specified percentage.
    :param image: Input image.
    :param scale_percent: Percentage by which the image should be resized.
    :return: Resized image.
    """
    width = int(image.shape[1] * scaleFactor)
    height = int(image.shape[0] * scaleFactor)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def extract_frames(video_path, timeframes, output_folder):
    """ Get images from video"""
    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the video.")
        return

    # Get the frames per second of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    newImagesFilePathArray = []

    for time in timeframes:
        # Calculate the frame number. Time is given in seconds
        frame_no = int(time * fps)

        # Set the video position to the frame we want to capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        # Read the frame
        ret, frame = cap.read()

        # If the frame was successfully read, save it
        if ret:
            # Downscale the image
            scaled_frame = resize_image(frame, IMAGE_SCALE_DOWN_FACTOR)  # reduce size

            output_path = f"{output_folder}/{time}_seconds.jpg"
            newImagesFilePathArray.append(output_path)
            cv2.imwrite(output_path, scaled_frame)
        else:
            print(f"Error: Couldn't extract frame at {time} seconds.")

    # Release the video capture object
    cap.release()
    return newImagesFilePathArray
    

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
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    # img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    # Canny Edge Detection. 

    img_blur = cv2.GaussianBlur(img, (5,5), 0)

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

    # shape[1]: width, shape[0]: height
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    blended = cv2.addWeighted(img1, alpha, img2_resized, 1-alpha, 0)

    return blended


def find_best_match(image1Edges, image2Edges, kernel_size, image1, mode="simple"):
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
    if mode == "extended":
        cv2.imshow("Position image", position_image)

    return image1

def addColor(img, color):
    # need to convert it to float32
    img = np.float32(img)

    # creating a mask for white regions (simpler than color extraction)
    _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # convert image to BGR
    colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    colored[mask>0] = color

    return colored

def generateDepthFromTwoImages(img1Path, img2Path, imgCropStartX, imgCropStartY, imgCropWidth, imgCropHeight, mode="simple"):
    # Swap images 1 and 2 for visualization purposes
    img1 = initilizeImage(img2Path, imgCropStartX, imgCropStartY, imgCropWidth, imgCropHeight)
    img2 = initilizeImage(img1Path, imgCropStartX, imgCropStartY, imgCropWidth, imgCropHeight)
    
    if mode == "extended":
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
    blended_initial = superimpose(img1, img2, 0.3)
    imageWithLines = convolute(img1Edges, img2Edges, KERNEL_SIZE, blended_initial)
    if mode == "extended":
        cv2.namedWindow('Resized Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Canny Edge Detection', imagesEdges)
        cv2.waitKey(0)
        cv2.imshow("Superimposed images", blended_initial)
        cv2.waitKey(0)
    
    return imageWithLines
        

def generateDepthFromVideo():
    ''' Using the video from the video folder, calculate depth'''
    # These are the timeframes for testVideo.mov in seconds
    timeframes = [1, 10, 17, 23, 30, 40, 47, 51, 61, 70, 76]
    # Get the frames from the video and store them in imagesFromVideo folder locally
    newImagesFilePathArray = extract_frames(VIDEO_PATH, timeframes, IMAGES_FROM_VIDEO_PATH)
    for i in range(1, len(newImagesFilePathArray)):
        imageWithLines = generateDepthFromTwoImages(newImagesFilePathArray[i-1], newImagesFilePathArray[i], IMG_CROP_START_X, IMG_CROP_START_Y, IMG_CROP_WIDTH, IMG_CROP_HEIGHT)
        cv2.imshow('Image with lines', imageWithLines)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # generateDepthFromTwoImages(IMG1_PATH, IMG2_PATH, IMG_CROP_START_X, IMG_CROP_START_Y, IMG_CROP_WIDTH, IMG_CROP_HEIGHT)
    generateDepthFromVideo()

if __name__ == "__main__":
    main()