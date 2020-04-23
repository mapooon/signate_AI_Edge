# -*- coding: utf-8 -*-

import numpy as np
import os
import json
import argparse
from PIL import Image



def make_json(annotations_path, categories):
    from collections import defaultdict
    from pathlib import Path
    from tqdm import tqdm

    json_data = {
        str(path).split('/')[-1].replace('png', 'jpg'): {
            c: defaultdict(list) for c in categories.keys()
        } for path in Path(annotations_path).glob('**/*.png')
    }
    for path, name in tqdm(zip(Path(annotations_path).glob('**/*.png'), json_data.keys())):
        img = Image.open(str(path)).convert('RGB')
        for c, val in categories.items():
            xs, ys = np.where((np.array(img) == val).all(axis=-1))
            category_pix = defaultdict(list)
            for x, y in zip(xs, ys):
                category_pix[x].append(y)
            for x, ys in category_pix.items():
                ys = np.array(ys)
                slided_ys = np.roll(ys, shift=1) + 1
                cut_points = np.where(ys != slided_ys)[0]
                for i, j in zip(cut_points, np.roll(cut_points, shift=-1) - 1):
                    json_data[name][c][str(x)].append(ys[[i, j]].tolist())

    return json_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',dest='seed',type=int)
    parser.add_argument('-e',dest='ensemble',action='store_true')
    parsed = parser.parse_args()
    current=os.getcwd()
    if parsed.ensemble:
        path_to_annotations=os.path.join(current,"predict/ensemble")
    
    else:    
        ip=parsed.seed
        path_to_annotations=os.path.join(current,"predict/ip_"+str(ip))
    
    categories = {'car':[0,0,255],'pedestrian':[255,0,0],'lane':[69,47,142],'signal':[255,255,0]}
    
    json_data = make_json(path_to_annotations, categories)
    with open('submit.json', 'w') as f:
        json.dump(json_data,f, sort_keys=True,separators=(',', ':'))
    