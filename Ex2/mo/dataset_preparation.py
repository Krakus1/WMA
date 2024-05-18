'''Przygotowuje obrazy datasetu do trenowania modelu deep learning

Returns:
    None: None
'''
import os
import argparse
import io
import csv
from PIL import Image
import cairosvg

def convert_image(old_path: str,
                  new_path: str,
                  output_format: str,
                  output_mode: str,
                  resolution: list[int]) -> None:
    '''Zamienia pliki .svg na .png lub .jpg
       Zmienia kodowanie na L/P/RGB/RGBA w zależności od wyboru
       Zmienia rozdzielczośc pliku

    Args:
        old_path (str): Ścieżka do istniejacego datasetu
        new_path (str): Ścieżla docelowa dla nowego datasetu
        output_format (str): Typ pliku: png lub jpg
        output_mode (str): Kodowanie koloru
        resolution (list[int, int]): Rozdzielczość nowego obrazu

    Returns:
       None: None
    '''

    # Pobierz rozszerzenie pliku
    _, ext = os.path.splitext(os.path.basename(old_path))

    if ext.lower() == '.svg':
        buffer = cairosvg.svg2png(file_obj=open(file=old_path, mode='rb'))
        image = Image.open(io.BytesIO(buffer))

    else:
        image = Image.open(old_path)

    image.convert(output_mode).resize(tuple(resolution)).save(new_path, output_format.upper())

def rename_bad_filenames(root: str, filename: str) -> str:
    '''Zamienia nazwe pliku z kilkuczłonowego na dwuczłonowy np. z abc.png.123 na abc.png

    Args:
        root (str): Ścieżka do pliku
        filename (str): Nazwa pliku
    Returns:
        str: Naprawioną nazwę pliku
    '''
    filename_parts = filename.split('.')

    # Jeśli nazwa pliku ma inną strukturę niż nazwa.rozszerzenie
    if len(filename_parts) != 2:
        # Wyszukaj pozycję jpg lub png w nazwie pliku
        index = 0
        if 'jpg' in filename_parts:
            index = filename_parts.index('jpg')
        elif 'png' in filename_parts:
            index = filename_parts.index('png')

        # Stworz nową nazwę z istniejących członów starej nazwy pliku
        new_filename = ''
        if index == 0:
            new_filename = '.'.join([filename_parts[1], filename_parts[0]])
        else:
            new_filename = '.'.join([filename_parts[0], filename_parts[index]])

        return new_filename
    return filename

def set_new_root(root: str, new_path: str) -> str:
    '''Ustanawia nową ścieżkę dla przetworzonego pliku

    Args:
        root (str): Ścieżka do oryginalnego pliku
        new_path (str): Ścieżka podana przez użytkownika

    Returns:
        str: Ścieżka do przetworzonego pliku
    '''

    path = os.path.normpath(root)
    path_split = path.split(os.sep)
    poke_index = path_split.index('pokemons')
    p_1 = '/'.join(path_split[poke_index:])
    pfull = '/'.join([new_path, p_1])

    return pfull

def save_csv(summary: dict[str, dict[str, int]]) -> None:
    '''Zapisuje Kategorie i liczbę przetworzonych/odrzuconych plików do .csv

    Args:
        summary (dict[str, dict[str, int]]): Dict z danymi dot. przetworzonych pllików
    '''
    print(summary)
    with open(file='output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Category', 'Processed', 'Rejected'])
        for category, values in summary.items():
            writer.writerow([category, values['processed'], values['rejected']])

def convert_dataset(args: argparse.Namespace) -> None:
    '''Główna funkcja, pobiera parametry od użytkownika i przetwarza pliki

    Args:
        args (argparse.Namespace): parametry programu podane podczas jego uruchamiania
    '''
    summary = {}
    for root, _, files in os.walk(args.input):
        for index, filename in enumerate(sorted(files), start=1):

            # Popraw pliki o niepoprawnej nazwie np nazwa.rozszrzenie.smieci na nazwa.rozszerzenie
            new_filename = rename_bad_filenames(root, filename)

            # Nazwa folderu gdzie zapisujemy nowe obrazki
            new_root = set_new_root(root, args.output)

            # Stworz docelowy folder jeśli nie istnieje
            if not os.path.exists(new_root):
                os.makedirs(new_root)

            # Zamiana nazwy pliku np Abra.jpg na liczbę np 003.jpg
            new_filename = f'{index:03d}.' + args.format

            # Scieżka nowego pliku
            new_file_path = os.path.join(new_root, new_filename)

            # Sciezka starego pliku
            old_file_path = os.path.join(root, filename)

            # Znajdź nazwę kategorii
            path = os.path.normpath(root)
            category = path.split(os.sep)[-1]

            # Dict do zapisu ilości przetworzonych/odrzuconych plików
            if category not in summary:
                summary[category] = {'processed': 0, 'rejected': 0}

            try:
                # Przetwarzanie obrazów
                convert_image(old_file_path,
                                new_file_path,
                                args.format,
                                args.mode,
                                args.resolution)

                # Podnieś ilośc przetworzonych plików dla kategorii
                summary[category]['processed'] += 1
            except Exception as e:
                print(f'error: {e}')
                # Podnieś ilośc odrzuconych plików dla kategorii
                summary[category]['rejected'] += 1

    save_csv(summary)

#--------------------------------------------------------------------------------------------------
#                                          MAIN I ARGPARSE
#--------------------------------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    '''Parser parametrów programu

    Returns:
        argparse.Namespace: Przechowuje arugmenty podane przez użytkownika
    '''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f',
                        '--format',
                        choices=['jpg', 'png'],
                        type=str,
                        required=True,
                        help='Final format of transformed dataset. jpg | png')

    parser.add_argument('-i',
                        '--input',
                        type=str,
                        required=True,
                        help='Input path to the original dataset.')

    parser.add_argument('-o',
                        '--output',
                        type=str,
                        required=True,
                        help='Output path to the transformed dataset.')

    parser.add_argument('-m',
                        '--mode',
                        type=str,
                        nargs='?',
                        choices=['L', 'P', 'RGB', 'RGBA'],
                        default='RGB',
                        help='Image mode of the transformed files. L | P | RGB | RGBA')

    parser.add_argument('-r',
                        '--resolution',
                        type=int,
                        required=True,
                        nargs=2,
                        help='New resolution of the images.')

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    '''Główna funkcja

    Args:
        args (argparse.Namespace): argumenty programu
    '''
    if args.format == 'jpg' and args.mode == 'RGBA':
        print('Unsupported mode RGBA for JPEG image')
        return

    convert_dataset(args)

if __name__ == '__main__':
    main(parse_arguments())
