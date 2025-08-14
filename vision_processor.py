import cv2
import numpy as np

class VisionProcessor:
    """A collection of static methods for image processing tasks."""

    @staticmethod
    def preprocess_with_clahe(img: np.ndarray) -> np.ndarray:
        """Applies CLAHE for contrast enhancement."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def align_images(template: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Aligns the live image to the template using template matching."""
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        x, y = max_loc
        h, w = template.shape
        aligned = image[y:y + h, x:x + w]
        if aligned.shape != template.shape:
            return image[:h, :w]  # Fallback if alignment fails
        return aligned

    @staticmethod
    def pixel_diff_change(ref: np.ndarray, live: np.ndarray, threshold: int = 25) -> tuple[bool, float]:
        """Compares two images and returns True if the difference exceeds a threshold."""
        diff = cv2.absdiff(ref, live)
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        diff_pixels = cv2.countNonZero(binary)
        total_pixels = binary.size
        if total_pixels == 0:
            return False, 0.0
        ratio = diff_pixels / total_pixels
        return ratio > 0.02, ratio  # 2% pixel change threshold

