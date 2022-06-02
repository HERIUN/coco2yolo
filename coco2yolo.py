import os, glob
from os import walk
from pycocotools.coco import COCO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import requests

# ann_path = 'filtered_4_class_val.json'
# img_dir = 'images/filter_4_val2017'
# task_ann_dir = 'labels/filter_4_val2017'

def main(args):

    ann_path = args.input_json
    img_dir = args.image_output_dir
    task_ann_dir = args.label_output_dir

    coco = COCO(ann_path)
    names = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]
    print(f"selected ann.json's category : \n{names}\n")

    name_correspond_table = {}
    cat_ids = []
    for cat_name in names:
        cat_id = coco.getCatIds(catNms=cat_name)[0]
        name_correspond_table[cat_name] = cat_id
        cat_ids.append(cat_id)
    print(name_correspond_table)
    print(cat_ids)

    id_correspond_dict = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

    imgIds = list(set().union(*[coco.getImgIds(catIds=cat_id) for cat_id in cat_ids]))
    print('count of images contain your filtered ann.json :', len(imgIds)) # count of images contain your filtered ann.json


    imgINFOs = coco.loadImgs(imgIds)

    filenames = os.listdir(img_dir)
    exist_img_infos = []
    noexist_img_infos = []

    for img_info in imgINFOs:
        if img_info['file_name'] in filenames:
            exist_img_infos.append(img_info)
        else:
            noexist_img_infos.append(img_info)
    print("exist_img_files : ", len(exist_img_infos))
    print("noexist_img_files : ",len(noexist_img_infos))


    # To download, url,names prepared
    img_datas_url = []
    img_names = []
    for img_info in tqdm(noexist_img_infos):
        img_datas_url.append(img_info['coco_url'])
        img_names.append(img_info['file_name'])
    url_names = list(zip(img_datas_url, img_names))

    def download_img_from_url(url_names):
        with open(f'{img_dir}/{url_names[1]}', 'wb') as f:
            f.write(requests.get(url_names[0]).content)

    with ThreadPoolExecutor(max_workers=None) as executor:
        executor.map(download_img_from_url, url_names)
    exist_img_infos.extend(noexist_img_infos) # 마지막에 리스트 확장


    # Truncates numbers to N decimals
    def truncate(n, decimals=0):
        multiplier = 10**decimals
        return int(n * multiplier) / multiplier

    for img_info in imgINFOs:
        dw,dh = 1.0/img_info['width'], 1.0/img_info['height']
        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(annIds)
        
        with open(f'{task_ann_dir}/{img_info["file_name"].replace(".jpg", ".txt")}', 'w') as f:
            for i in range(len(anns)):
                xmin = anns[i]['bbox'][0]
                ymin = anns[i]['bbox'][1]
                xmax = anns[i]['bbox'][2] + anns[i]['bbox'][0]
                ymax = anns[i]['bbox'][3] + anns[i]['bbox'][1]

                x = (xmin + xmax) / 2
                y = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin

                x *= dw
                w *= dw
                y *= dh
                h *= dh

                f.write(
                    f'{id_correspond_dict[anns[i]["category_id"]]} {truncate(x, 7)} {truncate(y, 7)} {truncate(w, 7)} {truncate(h, 7)}\n'
                )


    with open(img_dir.split('/')[-1]+'.txt', 'w') as f:
        names = os.listdir(img_dir)
        for name in names:
            f.write('./'+img_dir+os.sep+name+'\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="coco ann file to images, labels")
    
    parser.add_argument("-i", "--input_json", dest="input_json",
        help="path to a json file in coco format")
    parser.add_argument("-o", "--image_output_dir", dest="image_output_dir",
        help="path to save the images from input json")
    parser.add_argument("-c", "--label_output_dir", nargs='+', dest="label_output_dir",
        help="path to save the labels(.txt) from input json")

    args = parser.parse_args()

    main(args)