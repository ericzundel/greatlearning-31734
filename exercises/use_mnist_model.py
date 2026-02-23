import argparse
import os
from typing import Sequence

import cv2
from cv2.typing import MatLike
from matplotlib.pylab import ndarray
import numpy as np
import tensorflow as tf

DEBUG = False
NOISE_THRESHOLD=30
MODEL_INPUT_SHAPE=(28,28)
SCALED_INPUT_SHAPE=(20,20)

def non_noise_contours(contours: Sequence[ndarray]):
    """ Only return contours larger than the noise threshold in width + height"""
    result = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > NOISE_THRESHOLD and h > NOISE_THRESHOLD:
            result.append(contour)
    return result

def erode_contours(img: MatLike, x_offset: int, letters: list[tuple[int, ndarray]]):
    """ A processing step that attempts to separate letters, but fails miserably"""

    orig_width, orig_height = img.shape

    for i in range(3):
        # Create a small 2x2 or 3x3 kernel ... or larger
        # Tried 2x2, 3x3, 10x10, nothing worked with my sample image
        kernel = np.ones((10, 10), np.uint8)

        # Erode the image to thin the lines
        thinner_thresh = cv2.erode(img, kernel, iterations=1)
        if DEBUG:
            cv2.imshow(f"eroded piece loop{i}", thinner_thresh)
            cv2.waitKey()

        # Now find contours on 'thinner_thresh' instead of 'thresh'
        eroded_contours, _ = cv2.findContours(thinner_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        eroded_contours = non_noise_contours(eroded_contours)

        # Pull out the width and height from the contours into a list
        all_bounds = map(lambda rect: (cv2.boundingRect(rect)[2], cv2.boundingRect(rect)[3]), eroded_contours)

        if len(eroded_contours) > 1 and (orig_width,orig_height) not in all_bounds:

            for cnt in eroded_contours:
                x, y, w, h = cv2.boundingRect(cnt)

                roi = thinner_thresh[y:y+h, x:x+w]
                if DEBUG:
                    cv2.imshow("adding split contour", roi)
                    cv2.waitKey()
                letters.append((x_offset + x, roi)) # Store X coordinate to sort them later
            return
        else:
            # erode again!
            img = thinner_thresh
    # Couldn't erode. Add the full image passed in
    letters.append((x_offset, img))


def process_watershed(img: ndarray) -> ndarray:
    """ This pre-processing step finds boundaries between letters that might be overlapping."""

    # Assuming 'thresh' is your binary image (white text, black background)
    # 1. Clean up noise first
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)

    # 2. Distance Transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # 3. Threshold the distance to get the "sure foreground" (centers of letters)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 4. Dilate to find sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 5. Finding unknown region (Area where we aren't sure if it's letter or background)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Label the seeds
    ret, markers = cv2.connectedComponents(sure_fg)

    # 7. Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # 8. Mark the unknown region with 0
    markers[unknown == 255] = 0
    # 9. Apply the algorithm
    # Note: Watershed requires the original 3-channel image, so convert 'opening' to BGR
    img_bgr = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_bgr, markers)

    # 10. The boundary lines are marked with -1 in the markers array
    # Let's turn those boundaries into a mask to separate the letters

    # this makes a 1 pixel separation - not enough for the contours algorithm!
    #img[markers == -1] = 0

    # Markers is a 2d array (the image)
    # marker is a 1d array (one row in the image)
    y = 0
    for row in markers:
        x = 0
        for pixel in row:
            # each time we find a pixel set to -1, blot out that pixel and
            # the adjacent ones (unless we are at the beginning or end of the row)
            if pixel == -1:
                img[y][x] = 0
                if x > 0:
                    img[y][x-1] = 0
                if x < len(row) -1:
                    img[y][x+1] = 0
            x +=1
        y += 1
    return img

def split_by_contours(img: ndarray, x_offset: int, try_watershed: bool):

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    character_list: list[tuple[int, ndarray]] = []
    for cnt in non_noise_contours(contours):
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(cnt)
        region_of_interest = img[y:y+h, x:x+w].copy()

        if try_watershed and float(w)/h > 1.1:
            # Try using watershed to separate the characters
            watershed_img = process_watershed(region_of_interest)
            # Call this same fuction to try to split this image, but don't try to use
            # watershed again.
            sublist = split_by_contours(watershed_img, x, False)
            debug_images(sublist, "sublist")
            count = 0
            for img2 in sublist:
                character_list.append((x_offset + x + count, img2))
                count += 1
        else:
            # Save this sub-image. Each element is (postion, image)
            character_list.append((x + x_offset, region_of_interest))

    # Sort characters from left-to-right (contours are usually found in random order)
    character_list.sort(key=lambda x: x[0])

    # Return just the images now that they are in order
    character_list = list(map(lambda x: x[1], character_list))
    debug_images(character_list, "character")
    return character_list


def separate_characters(img: ndarray) -> Sequence[ndarray]:
    return split_by_contours(img, 0, True)

def resize_and_center_image(img):
    height, width = img.shape
    # Prepare a black image to copy the resized shape into
    canvas = np.zeros(MODEL_INPUT_SHAPE)

    # Calculate the proportionate new size of the scaled image
    # e.g. scale it down to fit in a 20x20 rectangle.  There is a
    # 4 pixel boundary on all sides for the mnist data set.
    new_size = SCALED_INPUT_SHAPE[0]
    scale = new_size / max(height, width)
    new_height, new_width = int(height * scale), int(width * scale)
    x_offset = int((new_size - new_width) / 2) + 4
    y_offset = int((new_size - new_height) / 2) + 4

    # resize the image, keeping the proportion
    #resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # copy the resized image to the center of the new black image
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    return canvas

def resize_and_center_images(image_list) -> list[ndarray]:
    return list(map(lambda img: resize_and_center_image(img), image_list))

def thin_images1(image_list) -> list[ndarray]:
    result = []
    for img in image_list:
        thinned = cv2.ximgproc.thinning(img)
        cv2.imshow("thinned", thinned)
        kernel = np.ones((2,2), np.uint8)
        thicker = cv2.dilate(thinned, kernel, iterations=1)
        cv2.imshow("thicker", thicker)
        cv2.waitKey()
        result.append(thicker)
    return result

def thin_images2(image_list) -> list[ndarray]:
    result = []
    for img in image_list:
        kernel = np.ones((10,10), np.uint8) # Use a larger kernel if the original image is high-res
        thinner = cv2.erode(img, kernel, iterations=2)
        result.append(thinner)
    return result

def blur_images(image_list, radius) -> list[ndarray]:
    return list(map(lambda img: cv2.GaussianBlur(img, radius, 0), image_list))


def outline_images(image_list) -> list[ndarray]:
    kernel = np.ones((3,3), np.uint8)
    return list(map(lambda img: cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel), image_list))


def debug_images(image_list, prefix):
    if DEBUG:
        count = 1
        for img in image_list:
            cv2.imshow(f"{prefix} {count}", img)
            count +=1
        cv2.waitKey()

# Workaround for 'QFontDatabase: Cannot find font directory...'
# Tell the Qt engine inside OpenCV where to find system fonts
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts'

def extract_digit_images(img):

    # Turn the image into black and white
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Look at the output (it looks good)
    debug_images([img], "threshold")

    # We should probably only watershed after a piece of the image shows up wider than it
    # should be.
    #img = process_watershed(img)
    #debug_images([img], "watershed")

    # Separate the image by finding the shape of the letters:
    # Find coordinates of all shapes
    imgs = separate_characters(img)

    # thin out the characters?
    #imgs = thin_images2(imgs)

    #imgs = outline_images(imgs)
    #debug_images(imgs, "outlined")

    #imgs = blur_images(imgs, (31, 51))
    #debug_images(imgs, "blurred 21x21")

    # Rezize them to 20x20 and center in a 28x28 square
    imgs = resize_and_center_images(imgs)
    debug_images(imgs, "resized")

    #imgs = blur_images(imgs, (3,3))
    #debug_images(imgs, "blurred 3x3")

    # Set image pixel values from ints 0-155  to floats ranging from 0.0-1.0
    imgs = np.array(imgs) / 255.0
    return imgs


def main():
    global DEBUG
    parser = argparse.ArgumentParser(
        description="Run a trained MNIST model on an image to recognize digits."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="data/zip_code_test1.jpg",
        help="Path to the input image file (default: data/zip_code_test1.jpg)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to display intermediate images",
    )
    args = parser.parse_args()
    DEBUG = args.debug

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    digit_images = extract_digit_images(img)

    if len(digit_images) == 0:
        print("*** Couldn't extract images", file=sys.stderr)
        return

    model = tf.keras.models.load_model("models/mnist_20_epochs.keras")
    predictions = model.predict(digit_images)

    result = []
    string_output = ""
    for prediction in predictions:
        #print("prediction: ", prediction)
        digit_val = np.argmax(tf.nn.softmax(prediction))
        #print (f"I pick {digit_val}")
        result.append(digit_val)
        string_output = string_output + str(int(digit_val))


    print(string_output)

main()

