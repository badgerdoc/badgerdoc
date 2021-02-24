from table_extractor.model.table import Cell
from table_extractor.tesseract_service.tesseract_extractor import TextExtractor
from os import listdir
from os.path import isfile, join, isdir
from datetime import datetime
import json
import pandas as pd
import asyncio
from typing import Tuple, List
import concurrent.futures
import itertools
from pprint import pprint
dir_path = '/home/ilia/gp_set/'


async def get_path(url):
    return 'annotated_dataset/' + url.split('/')[-3][:-4] + '/images/' + url.split('/')[-1]


def get_json_data(filename):
    with open(filename, 'r') as f:
        data = json.loads(f.read())
    return data


async def get_text_from_one_cell(test, ann):
    return test.extract(ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3])


async def make_annotations(ann, test):
    if ann['category_id'] == 4:
        header = Cell(ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] +
                      ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3])
        return header, 0
    elif ann['category_id'] == 2:
        text = await get_text_from_one_cell(test, ann)
        if text[0]:
            text = text[0].replace('\n', ' ')
            # print("Cell: ", text)
            cell = Cell(ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2],
                        ann['bbox'][1] + ann['bbox'][3], text_boxes=text)
            # print(cell.text_boxes)
            return cell, 1
    return 2, 2


async def collect_annotations(annotations, test):
    return await asyncio.gather(*[make_annotations(annotation, test)
                     for annotation in annotations])


async def get_cells_from_image(url, data) -> Tuple[List[Cell], List[Cell]]:
    print(url)
    image_id = [image for image in data['images'] if image['file_name'] in url]
    print(image_id)
    if image_id:
        test = TextExtractor(url)
        header_cells = []
        just_cells = []
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id[0]['id']]

        annotations_agg = await collect_annotations(annotations, test)
        print(annotations_agg)

        headers = [i[0] for i in annotations_agg if i[1] == 0]
        cells = [i[0] for i in annotations_agg if i[1] == 1]

        if headers:
            just_cells = []
            for header in headers:
                for each_cell in cells:
                    if each_cell.box_is_inside_another(header):
                        header_cells.append(each_cell)

        if cells:
            for cell in cells:
                if cell not in header_cells:
                    just_cells.append(cell)

        return header_cells, just_cells
            #print("Headers: ", len(headers))
            #print("Cells: ", len(cells))
            #print("Cells in headers: ", header_cells)


        # try:
        #     with open('headers.txt', 'a') as f:
        #         for header_cell in header_cells:
        #             f.write(header_cell.text_boxes + '\n')
        # except UnboundLocalError:
        #     print("Without headers: ", url)
        # try:
        #     with open('cells.txt', 'a') as f:
        #         for just_cell in just_cells:
        #             f.write(just_cell.text_boxes + '\n')
        # except UnboundLocalError:
        #     print('Without cells: ', url)

async def collect_text(onlyfiles, data):
    headers = []
    cells = []
    for filename in onlyfiles:
        result = await get_cells_from_image(filename, data)
        if result:
            file_headers, file_cells = result
            headers.extend(file_headers)
            cells.extend(file_cells)
    return headers, cells


async def files_collector(dir_name):
    images_dir = join(dir_name, 'images')
    onlyfiles = [join(images_dir, f) for f in listdir(images_dir) if isfile(join(images_dir, f))]
    json_path = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]
    with open(json_path[0], 'r') as f:
        data = json.load(f)

    return await collect_text(onlyfiles, data)


async def start():

    dirs = [join(dir_path, f) for f in listdir(dir_path) if isdir(join(dir_path, f))]

    result = await asyncio.gather(*[files_collector(dirr) for dirr in dirs])

    hs = [i[0] for i in result]
    cs = [i[1] for i in result]

    hs = itertools.chain(*hs)
    cs = itertools.chain(*cs)

    df1 = pd.Series([h.text_boxes for h in hs])
    df2 = pd.Series([c.text_boxes for c in cs])

    print(df2.count())
    df2 = df2[~df2.index.isin(df1)]
    print(df2.count())

    df1.to_csv('headers.txt', sep='\n', index=False)
    df2.to_csv('cells.txt', sep='\n', index=False)

    # df1.drop(df1[df1['count'] < 3].index, inplace=True)
    # df2.drop(df2[df2['count'] < 3].index, inplace=True)
    # print(df1, df2)


def main():
    start_time = datetime.now()
    asyncio.run(start())
    end_time = datetime.now() - start_time
    print(f'Program ends in {end_time}')


if __name__ == '__main__':
    main()
