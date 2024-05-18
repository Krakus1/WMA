#--------------------------------------------------------------------------------------------------
#                                          IMPORTS
#--------------------------------------------------------------------------------------------------
import argparse
import os
import csv
import io
import cairosvg
from PIL import Image

def save_summary(summary: dict[str, dict[str, int]]) -> None:
    """Save summary data

    Args:
        summary (dict[str, dict[str, int]]): dictionary with data
    """
    #print(summary)
    with open(file='summary.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Category', 'Done', 'Err'])
        for category, values in summary.items():
            writer.writerow([category, values['done'], values['err']])

def correct_img(old_path: str,
                  new_path: str,
                  output_format: str,
                  output_mode: str,
                  resolution: list[int]) -> None:
    """Correct image 

    Args:
        old_path (str): actual file path
        new_path (str): new file path
        output_format (str): format file
        output_mode (str): mode file
        resolution (list[int]): resolution new file
    """
    _, ext = os.path.splitext(os.path.basename(old_path))

    if ext.lower() == '.svg':
        with open(file=old_path, mode='rb') as file_obj:
            buffer = cairosvg.svg2png(file_obj=file_obj)
        image = Image.open(io.BytesIO(buffer))

    else:
        image = Image.open(old_path)

    image.convert(output_mode).resize(tuple(resolution)).save(new_path, output_format.upper())

def correct_filenames(filename: str) -> str:
    """ function to correct name file

    Args:
        root (str): path from set file
        filename (str): destination new file

    Returns:
        str: new file name
    """
    filename_p = filename.split('.')

    if len(filename_p) != 2:
        index = 0
        if 'jpg' in filename_p:
            index = filename_p.index('jpg')
        elif 'png' in filename_p:
            index = filename_p.index('png')

        new_filename = ''
        if index == 0:
            new_filename = '.'.join([filename_p[1], filename_p[0]])
        else:
            new_filename = '.'.join([filename_p[0], filename_p[index]])

        return new_filename
    return filename

def create_new_folder(root: str, new_path: str) -> str:
    """function to create folder for files

    Args:
        root (str): main folder
        new_path (str): new path to folder

    Returns:
        str: new_folder_name
    """

    path = os.path.normpath(root)
    path_split = path.split(os.sep)
    folder_p = '/'.join(path_split[path_split.index('pokemons'):])
    new_folder_name = '/'.join([new_path, folder_p])

    return new_folder_name

def organize_files(args: argparse.Namespace) -> None:
    """main function, create new folder, new name, correct image

    Args:
        args (argparse.Namespace): param users, input, output, extension, mode, resol
    """
    summary = {}
    for root, _, files in os.walk(args.input):
        for index, filename in enumerate(sorted(files), start=1):

            # Correct name file
            new_filename = correct_filenames(filename)

            # Folder to save new file
            new_folder = create_new_folder(root, args.output)
            new_filename = f'{index:03d}.' + args.ext

            # If not exists
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)

            new_file_path = os.path.join(new_folder, new_filename)
            old_file_path = os.path.join(root, filename)

            # Find category
            path = os.path.normpath(root)
            category = path.split(os.sep)[-1]

            # Dict do zapisu ilości przetworzonych/odrzuconych plików
            if category not in summary:
                summary[category] = {'done': 0, 'err': 0}

            try:
                # Correct image
                correct_img(old_file_path,
                                new_file_path,
                                args.ext,
                                args.mode,
                                args.resol)

                summary[category]['done'] += 1
            except Exception as e:
                print(f'error: {e}')
                summary[category]['err'] += 1

    save_summary(summary);

def par_arg() -> argparse.Namespace:
    """option application

    Returns:
        argparse.Namespace: option application
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        required=True,
                        help='Input path')

    parser.add_argument('-o',
                        '--output',
                        type=str,
                        required=True,
                        help='Output path')

    parser.add_argument('-e',
                        '--ext',
                        choices=['jpg', 'png'],
                        type=str,
                        required=True,
                        help='Final extension jpg | png')

    parser.add_argument('-m',
                        '--mode',
                        type=str,
                        nargs='?',
                        choices=['L', 'P', 'RGB', 'RGBA'],
                        default='RGB',
                        help='Image mode RGB | RGBA | L | P')

    parser.add_argument('-r',
                        '--resol',
                        type=int,
                        required=True,
                        nargs=2,
                        help='Resolution of the images')

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    """main

    Args:
        args (argparse.Namespace): option applications
    """
    if args.mode == 'RGBA' and args.ext == 'jpg' :
        print('Brak wsparcia')
        return

    organize_files(args)

if __name__ == '__main__':
    main(par_arg())
