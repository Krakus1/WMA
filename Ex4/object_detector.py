"""
Detector by ORB

Classes:
    ObjectDetector: Implements a simple convolutional neural network architecture.

Functions:
    def train(self, img: ArrayLike, present: bool) -> None:
    classify(self, img: ArrayLike) -> bool:
    load_image(path: str) -> np.ndarray:
    parse_arguments():

How to use the module:
    - Configure the paths.
    - Inside image to path.
"""

import argparse
import cv2
import numpy as np
from numpy.typing import ArrayLike

class ObjectDetector:
    """ObjectDetector"""
    def __init__(self, threshold: float):
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.threshold = threshold
        self.trained_descriptors: list[np.ndarray] = []
        self.object_present: list[bool] = []

    def train(self, img: ArrayLike, present: bool) -> None:
        """Train the detector with an image.

        Args:
            img (ArrayLike): Image to train on.
            present (bool): Whether the object is present in the image.
        """
        _, descriptors = self.orb.detectAndCompute(img, None)

        if descriptors is not None:
            self.trained_descriptors.append(descriptors)
            self.object_present.append(present)

    def classify(self, img: ArrayLike) -> bool:
        """Classify the image.

        Args:
            img (ArrayLike): Image to classify.

        Returns:
            bool: Whether the object is detected in the image.
        """
        _, descriptors = self.orb.detectAndCompute(img, None)

        if descriptors is None:
            return False

        total_score = 0.0

        for trained_desc, is_present in zip(self.trained_descriptors, self.object_present):
            if not is_present:
                continue

            matches = self.matcher.match(descriptors, trained_desc)

            for match in matches:
                total_score += 1 / (1 + match.distance)

        return total_score > self.threshold

def load_image(path: str) -> np.ndarray:
    """Load an image from a given path.

    Args:
        path (str): Path to the image file.

    Raises:
        ValueError: If the image could not be loaded.

    Returns:
        np.ndarray: Loaded image.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at path {path} could not be loaded")
    return img

def parse_arguments():
    """Parses command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Object Detector using ORB and BFMatcher')
    parser.add_argument('--path', type=str, required=True, help='Path to the test image')
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='Threshold for object detection')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    detector = ObjectDetector(threshold=args.threshold)

    train_image_with_object = load_image('train_image_with_object.jpg')
    train_image_without_object = load_image('train_image_without_object.jpg')

    detector.train(train_image_with_object, present=True)
    detector.train(train_image_without_object, present=False)

    test_image = load_image(args.path)

    result = detector.classify(test_image)
    print("Object present:", result)
