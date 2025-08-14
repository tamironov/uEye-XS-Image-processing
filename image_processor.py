# image_processor.py
import cv2
import numpy as np

class ImageProcessor:
    """
    A utility class for image processing operations.
    All methods are static as they operate on input images without needing instance state.
    """

    @staticmethod
    def preprocess_with_clahe(img):
        """
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
        Converts the image to grayscale first.
        :param img: Input BGR image (NumPy array).
        :return: Processed grayscale image.
        """
        if len(img.shape) == 3: # Check if it's a color image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else: # Assume it's already grayscale
            gray = img
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray)

    @staticmethod
    def align_images(template, image):
        """
        Aligns an 'image' to a 'template' using template matching.
        This is a simple alignment and assumes the template is present within the image.
        It returns the region of 'image' that best matches the 'template'.
        :param template: The reference image (grayscale, preprocessed).
        :param image: The live image (grayscale, preprocessed) to align.
        :return: The aligned region of the 'image' with the same dimensions as the 'template'.
                 If alignment fails or dimensions mismatch, it returns a cropped version of the image.
        """
        # Ensure both are grayscale
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        # Find the location of the best match
        _, _, _, max_loc = cv2.minMaxLoc(res)
        x, y = max_loc # Top-left corner of the matched region
        h, w = template.shape # Height and width of the template

        # Extract the aligned region from the image
        aligned = image[y:y+h, x:x+w]

        # Fallback if the aligned region doesn't match the template's expected shape
        # This can happen if the matched region goes out of bounds or is too small
        if aligned.shape != template.shape:
            # As a fallback, just return a crop of the live image to the template size
            # This might not be perfectly aligned but prevents errors.
            return image[:h, :w]
        return aligned

    @staticmethod
    def pixel_diff_change(ref, live, threshold=25, change_ratio_threshold=0.02):
        """
        Calculates the pixel difference between two grayscale images and determines if a change occurred.
        :param ref: Reference grayscale image.
        :param live: Live grayscale image.
        :param threshold: Pixel intensity difference threshold to consider a pixel "changed".
        :param change_ratio_threshold: Percentage of changed pixels to consider a significant change.
        :return: Tuple (bool: True if change detected, float: ratio of changed pixels).
        """
        # Ensure both images have the same dimensions for comparison
        if ref.shape != live.shape:
            # This should ideally not happen if alignment is robust, but for safety:
            min_h = min(ref.shape[0], live.shape[0])
            min_w = min(ref.shape[1], live.shape[1])
            ref = ref[:min_h, :min_w]
            live = live[:min_h, :min_w]
            if ref.size == 0 or live.size == 0: # Handle case where one dimension is zero
                return False, 0.0

        # Calculate absolute difference between images
        diff = cv2.absdiff(ref, live)
        # Apply a binary threshold to highlight significant differences
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        # Count non-zero pixels (pixels that changed above the threshold)
        diff_pixels = cv2.countNonZero(binary)
        # Calculate total pixels in the comparison area
        total_pixels = binary.size
        
        if total_pixels == 0: # Avoid division by zero
            return False, 0.0

        # Calculate the ratio of changed pixels
        ratio = diff_pixels / total_pixels
        # Determine if a significant change is detected
        change_detected = ratio > change_ratio_threshold
        return change_detected, ratio
