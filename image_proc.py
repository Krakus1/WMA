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

def blur_size(text:str)-> tuple[int,int]:
    try:
        blur = tuple(map(int(text.split(','))))
        if  len(blur) != 2 or blur[0]%2==0 or blur[1]%2==0:
            raise ValueError()
    except ValueError:
        print(f'Blur size must be given as INT,INT. Both ints must be odd. Value given {text}')
        raise argparse.ArgumentError()

def parse_arguments()-> argparse.Namespace:
    parser:argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, required=True, 
                        help='Path to image that will be processed')
    parser.add_argument('-b', '--blur_range', type=blur_size, default=3,
                        help='Size of blur')
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
            except cv2.error as e:
                print(f'Cv2 error -> {e}')
