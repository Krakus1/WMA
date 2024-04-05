import argparse
import cv2
import numpy as np

KEYCODE_ESC = 27
EXIT_KEYS = [ord('q'), KEYCODE_ESC]

def to_greyscale(img: np.ndarray)->np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def blur(img: np.ndarray)->np.ndarray:
    return cv2.blur(img,(5,5))

KEYBINDS = {
    ord('g'):to_greyscale,
    ord('b'):blur,    
}

def parse_arguments()-> argparse.Namespace:
    parser:argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, required=True, 
                        help='Path to image that will be processed')
    return parser.parse_args()


if __name__ == '__main__':
    args: argparse.Namespace = parse_arguments()
    print(type(args))
    img: np.ndarray = cv2.imread(args.image_path)
    
    halted = False
    while not halted: 
        cv2.imshow('Our image', img)
        keycode: int = cv2.waitKey()
        if keycode in EXIT_KEYS:
            halted = True
        else:
            try:
                img = KEYBINDS[keycode](img)
            except KeyError:
                print(f'Keycode {keycode} not supported')
