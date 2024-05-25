import glob
import argparse
import json
import io
import os
import lmdb
from PIL import Image
from tqdm import tqdm
import cv2
import shutil
import numpy as np

from pathlib import Path

def save_lmdb(env_path, meta_dict, transform=None):
    env_path = Path(env_path)  # Use pathlib
    data_dict = dict()  # Initialize outside the try block
    # Create parent directory (if not exists) before opening LMDB environment
    os.makedirs(env_path.parent, exist_ok=True) 
    try:
        env = lmdb.open(
            env_path.as_posix().encode('utf-8'),  
            map_size=1024 ** 2, 
            max_readers=100, 
            lock=False  
        )

        with env.begin(write=True) as txn:
            for font_name, font_meta in tqdm(meta_dict.items()):
                img_paths = font_meta['path']
                if font_name == "background":
                    continue
                if isinstance(font_meta['charlist'], dict):
                    all_chars = font_meta['charlist']['upper'] + font_meta['charlist']['lower']
                else:
                    all_chars = font_meta['charlist']
                for img_name in all_chars:
                    key = f"{font_name}_{img_name}"
                    img_path = os.path.join(img_paths, f"{img_name}.png").replace("\\", "/")  # Ensure forward slashes

                    if not os.path.exists(img_path):
                        continue  # Skip if image doesn't exist
                    img = Image.open(img_path).convert('RGB')

                    if transform is not None:
                        img = transform(img)
                    _, img_byte = cv2.imencode('.png', np.array(img))

                    # write lmdb
                    txn.put(key.encode(), img_byte)
                    if font_name not in data_dict.keys():
                        data_dict[font_name] = []
                    data_dict[font_name].append(img_name)

    except lmdb.Error as e:
        print(f"LMDB error: {e}")
        # Add your error handling logic here

    return data_dict



def getCharList(root):
    """[get all characters this font exists]

    Args:
        root (string): folder path

    Returns:
        [list]: char list
    """
    charlist = []
    for img_path in (glob.glob(root + '/*.jpg') + glob.glob(root + '/*.png')):
        ch = os.path.basename(img_path).split('.')[0]
        charlist.append(ch)
    return charlist


def getMetaDict(font_path_list):
    """[generate a dict to save the relationship between font and its existing characters]
    Args:
        font_path_list (List): [training fonts list]

    Returns:
        [dict]: [description]
    """
    meta_dict = dict()
    print("ttf_path_list:", len(font_path_list))
    for font_path in tqdm(font_path_list):
        font_name = os.path.basename(font_path)
        meta_dict[font_name] = {
            "path": font_path,
            "charlist": None
        }
        meta_dict[font_name]["charlist"] = getCharList(font_path)
    return meta_dict


def build_meta4train_lmdb(args):
    # saving directory
    out_dir = os.path.join(args.saving_dir, 'meta')
    lmdb_path = os.path.join(args.saving_dir, 'lmdb')
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    os.makedirs(lmdb_path, exist_ok=True)

    trainset_dict_path = os.path.join(out_dir, 'trainset_dict.json')
    # directory of your content_font
    content_font = args.content_font

    # ===================================================================#
    train_font_dir = args.train_font_dir
    validation_font_dir = args.val_font_dir

    dict_save_path = os.path.join(out_dir, "trainset_ori_meta.json")
    font_path_list = []

    font_chosen = []
    print(train_font_dir)
    for font_name in os.listdir(train_font_dir):
        # print(font_name)
        font_chosen.append(os.path.join(train_font_dir, font_name))

    font_chosen += glob.glob(validation_font_dir + "/*")
    font_chosen = list(set(font_chosen))

    print('num of fonts: ', len(font_chosen))

    # add content font
    if content_font not in font_chosen:
        font_chosen.append(content_font)

    out_dict = getMetaDict(font_chosen)
    with open(dict_save_path, 'w') as fout:
        json.dump(out_dict, fout, indent=4, ensure_ascii=False)

    valid_dict = save_lmdb(lmdb_path, out_dict)
    with open(trainset_dict_path, "w") as f:
        json.dump(valid_dict, f, indent=4, ensure_ascii=False)


def build_train_meta(args):
    train_meta_root = os.path.join(args.saving_dir, 'meta')
    # content
    content_font_name = os.path.basename(args.content_font)  # 'kaiti_xiantu'

    # ==============================================================================#
    # 保存train.json的路径
    save_path = os.path.join(train_meta_root, "train.json")
    meta_file = os.path.join(train_meta_root, "trainset_dict.json")

    with open(meta_file, 'r') as f_in:
        original_meta = json.load(f_in)
    with open(args.seen_unis_file) as f:
        seen_unis = json.load(f)
    with open(args.unseen_unis_file) as f:
        unseen_unis = json.load(f)

    # all font names
    all_style_fonts = list(original_meta.keys())

    unseen_ttf_dir = args.val_font_dir  # "/ssd1/tanglc/cvpr_image/cu_font_122_val"
    unseen_ttf_list = [os.path.basename(x) for x in glob.glob(unseen_ttf_dir + '/*')]
    unseen_style_fonts = [ttf for ttf in unseen_ttf_list]

    # get font in training set
    train_style_fonts = list(set(all_style_fonts) - set(unseen_style_fonts))

    train_dict = {
        "train": {},
        "avail": {},
        "valid": {}
    }

    for style_font in train_style_fonts:
        avail_unicodes = original_meta[style_font]
        train_unicodes = list(set.intersection(set(avail_unicodes), set(seen_unis)))
        train_dict["train"][style_font] = train_unicodes  # list(intersection_unis)

    for style_font in all_style_fonts:
        avail_unicodes = original_meta[style_font]
        train_dict["avail"][style_font] = avail_unicodes

    print("all_style_fonts:", len(all_style_fonts))
    print("train_style_fonts:", len(train_dict["train"]))
    print("val_style_fonts:", len(unseen_style_fonts))
    print("seen_unicodes: ", len(seen_unis))
    print("unseen_unicodes: ", len(unseen_unis))

    # validation set
    train_dict["valid"] = {
        "seen_fonts": list(train_dict["train"].keys()),
        "unseen_fonts": unseen_style_fonts,
        "seen_unis": seen_unis,
        "unseen_unis": unseen_unis,
    }

    with open(save_path, 'w') as fout:
        json.dump(train_dict, fout, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saving_dir", help="directory where your lmdb file will be saved")
    parser.add_argument("--content_font", help="root path of the content font images")
    parser.add_argument("--train_font_dir", help="root path of the training font images")
    parser.add_argument("--val_font_dir", help="root path of the validation font images")
    parser.add_argument("--seen_unis_file", help="json file of seen characters")
    parser.add_argument("--unseen_unis_file", help="json file of unseen characters")
    args = parser.parse_args()
    build_meta4train_lmdb(args)
    build_train_meta(args)


