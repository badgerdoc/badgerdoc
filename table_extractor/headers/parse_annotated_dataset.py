import json
import re
from os import listdir
from os.path import isfile, join, isdir

from table_extractor.model.table import Cell
from table_extractor.tesseract_service.tesseract_extractor import TextExtractor


def get_path(url):
    match = url[url.rfind('/')+1:]
    return match

def get_json_data(filename):
    with open(filename, 'r') as f:
        data = json.loads(f.read())
    return data

def get_text_from_one_cell(test, ann):
    return test.extract(ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3])

def get_cells_from_image(url, data):
    image_id = [image for image in data['images'] if get_path(image['path']) == get_path(url)]

    if image_id:
        test = TextExtractor(url)
        headers = list()
        cells = list()
        header_cells = list()
        just_cells = list()
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id[0]['id']]
        for ann in annotations:
            if ann['category_id'] == 15:
                headers.append(Cell(ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]))
            elif ann['category_id'] == 10:
                text = get_text_from_one_cell(test, ann)
                if text[0]:
                    text = text[0].replace('\n', ' ')
                    cell = Cell(ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3], text_boxes=text)
                    cells.append(cell)
        if headers:
            for header in headers:
                for each_cell in cells:
                    if each_cell.box_is_inside_another(header):
                        with open('headers.txt', 'a') as f:
                            f.write(each_cell.text_boxes + '\n')
                        header_cells.append(each_cell)
        if cells:
            for cell in cells:
                if cell not in header_cells:
                    with open('cells.txt', 'a') as f:
                        f.write(cell.text_boxes + '\n')


def main():
    dir_path = 'annotated_dataset/'
    dirs = [join(dir_path, f) for f in listdir(dir_path) if isdir(join(dir_path, f))]
    
    for dir_name in dirs:
        images_dir = join(dir_name, 'images')
        onlyfiles = [join(images_dir, f) for f in listdir(images_dir) if isfile(join(images_dir, f))]
        json_path = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]
    
        with open(json_path[0], 'r') as f:
            data = json.load(f)
        for filename in onlyfiles:
            get_cells_from_image(filename, data)


if __name__ == '__main__':
    main()
