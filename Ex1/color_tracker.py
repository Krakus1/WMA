'''
This is a simple example of color object tracking.
'''
import argparse
from enum import Enum
from typing import Any
import cv2
import numpy as np
from PIL import Image


class ProcessingType(Enum):
    '''
    klasa okreslajaca typy procesow jakie obsluguje skrypt
    '''
    RAW = 0
    TRACKER = 1
    HUE = 2
    SATURATION = 3
    VALUE = 4
    MASK = 5

# MODEL
class ColorTracker:
    '''
    Input:  video_path: sciezka do pliku;
            tracked_color: kolor w modelu HSV
            range: tolerancja dla kolorow
    Ouput: brak
    '''
    def __init__(self, video_path: str,
                 tracked_color: None | np.ndarray[Any, np.dtype[np.generic]] = None,
                 range_tolerance: tuple[int, int, int] = (10, 40, 40)) -> None:
        '''
            funkcja inicjujaca
        '''
        self._video = cv2.VideoCapture(video_path) # pylint: disable=no-member
        if not self._video.isOpened():
            raise ValueError(f'Unable to open video at path {video_path}.')


        self._tracked_color = tracked_color
        self._frame: cv2.Mat | np.ndarray[Any, np.dtype[np.generic]] # pylint: disable=no-member
        self._processed_frame: cv2.Mat | np.ndarray[Any, np.dtype[np.generic]] # pylint: disable=no-member
        self._processing_type: ProcessingType = ProcessingType.RAW
        self._range_tolerance = range_tolerance

    def set_reference_color_by_position(self, x: int, y: int) -> None:
        '''
        ustawienie wybranego koloru do sledzenia
        '''
        hsv_frame: cv2.Mat | np.ndarray[Any, np.dtype[np.generic]]=cv2.cvtColor( # pylint: disable=no-member
            self._frame, cv2.COLOR_BGR2HSV) # pylint: disable=no-member
        self._tracked_color = hsv_frame[y, x, :]

    def update_frame(self) -> bool:
        '''
        zapisanie klatki do procesowania
        '''
        read_successful, self._frame = self._video.read()
        if read_successful:
            self._process_frame()
        return read_successful

    def _calc_range_color(self) -> tuple[np.ndarray[Any, np.dtype[np.generic]],
                                         np.ndarray[Any, np.dtype[np.generic]]]:
        '''
        obliczenie zakresu
        output: dwie wartosci min i max zakresu
        '''

        if self._tracked_color is not None and self._range_tolerance is not None:
            upper_range = np.array([
                min(self._tracked_color[0] + self._range_tolerance[0], 179),
                min(self._tracked_color[1] + self._range_tolerance[1], 255),
                min(self._tracked_color[2] + self._range_tolerance[2], 255)
            ])

            lower_range = np.array([
                max(self._tracked_color[0] - self._range_tolerance[0], 0),
                max(self._tracked_color[1] - self._range_tolerance[1], 0),
                max(self._tracked_color[2] - self._range_tolerance[2], 0)
            ])
            return lower_range, upper_range

        return np.zeros(1), np.zeros(1)

    def _process_frame(self) -> None:
        '''
        snapshop - ustalenie typu
        policzenie macierzy otaczajacej srodkowy piksel
        RAW - original
        HUE - kolor
        SATURATION - nasycenie
        VALUE - jasnosc
        MASK - maska
        TRACKER - sledzenie
        '''
        hsv_frame: np.ndarray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2HSV) # pylint: disable=no-member

        if self._processing_type == ProcessingType.RAW:
            self._processed_frame = self._frame
            return

        if self._processing_type == ProcessingType.HUE:
            self._processed_frame = hsv_frame[:, :, 0]
            return

        if self._processing_type == ProcessingType.SATURATION:
            self._processed_frame = hsv_frame[:, :, 1]
            return

        if self._processing_type == ProcessingType.VALUE:
            self._processed_frame = hsv_frame[:, :, 2]
            return

        if self._tracked_color is None:
            raise ValueError(
                'Attempted processing mode that requires a tracking color set without it set.')

        lower_bound, upper_bound = self._calc_range_color()
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound) # pylint: disable=no-member

        if self._processing_type == ProcessingType.MASK:
            self._processed_frame = mask

        if self._processing_type == ProcessingType.TRACKER:
            mask_ = Image.fromarray(mask)
            bbox = mask_.getbbox()
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                self._processed_frame = cv2.rectangle(self._frame, # pylint: disable=no-member
                                                      (x1, y1), (x2, y2), (0, 255, 0), 5) # pylint: disable=no-member
            else:
                self._processed_frame = self._frame
            return

    def get_frame(self) -> np.ndarray:
        '''
        funkcja zwracjaca  snapshop
        '''
        if self._frame is None:
            raise ValueError('Attempted to get frame from uninitialized color tracker.')
        return self._frame.copy()

    def get_processed_frame(self) -> np.ndarray:
        '''
        funkcja zwracajaca kopie snapshop
        '''
        return self._processed_frame.copy()

    def set_processing_type(self, ptype: ProcessingType) -> None:
        '''
        ustawienie typu procesu
        '''
        self._processing_type = ptype


# VIEW
class Display:
    '''
    klasa odpowiedzialna za wyswielenie
    '''
    def __init__(self, window_name: str)->None:
        '''
        funkcja inicjujaca
        '''
        cv2.namedWindow(window_name) # pylint: disable=no-member
        self._window_name = window_name

    def update_display(self, image: np.ndarray)->None:
        '''
        wyswietlenie snapshot
        '''
        cv2.imshow(self._window_name, image) # pylint: disable=no-member

    def get_window_name(self)-> str:
        '''
        pobranie nazwy okna
        '''
        return self._window_name

# CONTROLER
class EventHandler:
    '''
    klasa odpowiedzialna za obsluge zdarzen
    '''
    PROCESSING_TYPE_KEYMAP = {
        ord('h'): ProcessingType.HUE,
        ord('s'): ProcessingType.SATURATION,
        ord('v'): ProcessingType.VALUE,
        ord('r'): ProcessingType.RAW,
        ord('m'): ProcessingType.MASK,
        ord('t'): ProcessingType.TRACKER
    }

    def __init__(self, tracker: ColorTracker, display: Display, timeout: int)->None:
        '''
        klasa inicjalizujaca
        '''
        self._window_name = display.get_window_name()
        self._tracker = tracker
        self._timeout = timeout

        cv2.setMouseCallback(self._window_name, self._handle_mouse) # pylint: disable=no-member


    def _handle_mouse(self, event: int, x: int, y: int, flags, param)->None:
        '''
        pobranie pozycji po kliknieciu myszki
        '''
        if event == cv2.EVENT_LBUTTONDOWN: # pylint: disable=no-member
            self._tracker.set_reference_color_by_position(x,y)

    def _handle_keys(self)-> bool:
        '''
        funkcja odpowiedzialna za obsluge skrotami klawiaturowymi
        '''
        keycode = cv2.waitKey(self._timeout) # pylint: disable=no-member

        if keycode == ord('q') or keycode == 27:
            return True
        if keycode in EventHandler.PROCESSING_TYPE_KEYMAP:
            self._tracker.set_processing_type(EventHandler.PROCESSING_TYPE_KEYMAP[keycode])
        return False

    def handle_events(self)-> bool:
        '''
        funkcja zwracajaca wartosc logiczna w zaleznosci od przycisku
        '''
        return self._handle_keys()


def parse_arguments()-> argparse.Namespace:
    '''
    funkcja przejmujaca przekazane argumenty
    '''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--video_path', type=str, required=True,
                        help='Path to video that will be processed.')
    parser.add_argument('-s', '--saturation', type=int, required=True,
                        help='Saturation range.')
    parser.add_argument('-a', '--value', type=int, required=True,
                        help='Value range.')
    parser.add_argument('-u', '--hue', type=int, required=True,
                        help='Hue range.')


    return parser.parse_args()
def main(args: argparse.Namespace)-> None:
    '''
    funkcja glowna
    '''
    try:
        window_name = 'Color tracker'
        waitkey_timeout = 10
        tracker = ColorTracker(video_path=args.video_path,
                                    range_tolerance=(args.hue, args.saturation, args.value))
        display = Display(window_name)
        event_handler = EventHandler(tracker, display, waitkey_timeout)
        while True:
            if not tracker.update_frame():
                break
            display.update_display(tracker.get_processed_frame())
            if event_handler.handle_events():
                break

    except ValueError as e:
        print(e)


if __name__ == '__main__':
    main(parse_arguments())
