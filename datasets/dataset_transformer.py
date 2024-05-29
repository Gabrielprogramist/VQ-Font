import torch
import json
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import os
import cv2 as cv
import logging
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CombTrainDataset(Dataset):
    """
    CombTrainDataset
    """
    def __init__(self, env, env_get, avails, all_content_json, content_font, transform=None):
        self.env = env
        self.env_get = env_get

        self.num_Positive_samples = 2 #一种font中取几个样本
        self.k_shot = 3

        with open(all_content_json, 'r') as f:
            self.all_characters = json.load(f)

        self.avails = avails #可用于训练的字体 data_meta["train"]格式为{"fontname1":[charlist],...,fontname2":[charlist]}
        self.unis = sorted(self.all_characters)
        self.fonts = list(self.avails) #获取所有训练字体的name
        self.n_fonts = len(self.fonts) #num of fonts(train)
        self.n_unis = len(self.unis)

        self.content_font = content_font
        self.transform = transform  #数据预处理方法

    def random_get_trg(self, avails, font_name):
        target_list = list(avails[font_name])
        trg_uni = np.random.choice(target_list, self.num_Positive_samples * 4)
        return [str(trg_uni[i]) for i in range(0, self.num_Positive_samples * 4)]

    def sample_pair_style(self, font, ref_unis):
        try:
            if len(ref_unis) < 3:
                logging.error(f"Not enough reference unis for font {font}: {ref_unis}")
                return None
            # logging.debug(f"Trying to sample pair style images for font {font} with unis {ref_unis}")
            imgs = torch.cat([self.env_get(self.env, font, uni, self.transform) for uni in ref_unis])
            # logging.debug(f"Sampled pair style images for font {font} with unis {ref_unis}: shape {imgs.shape}")
        except Exception as e:
            logging.error(f"Error sampling pair style images for font {font} with unis {ref_unis}: {e}")
            return None
        return imgs


    def __getitem__(self, index):
        font_idx = index % self.n_fonts
        font_name = self.fonts[font_idx]
        # logging.debug(f"Getting item for index {index}: font {font_name} (index {font_idx})")
        while True:
            style_unis = self.random_get_trg(self.avails, font_name)
            trg_unis = style_unis[:self.num_Positive_samples]
            sample_index = torch.tensor([index])
            avail_unis = self.avails[font_name]
            ref_unis = style_unis[self.num_Positive_samples:]
            style_imgs = torch.stack([self.sample_pair_style(font_name, ref_unis[i*3:(i+1)*3]) for i in range(0, self.num_Positive_samples)], 0)
            if style_imgs is None:
                continue
            trg_imgs = torch.stack([self.env_get(self.env, font_name, uni, self.transform) for uni in trg_unis], 0)
            trg_uni_ids = [self.unis.index(uni) for uni in trg_unis]
            font_idx = torch.tensor([font_idx])
            content_imgs = torch.stack([self.env_get(self.env, self.content_font, uni, self.transform) for uni in trg_unis], 0)
            ret = (
                torch.repeat_interleave(font_idx, style_imgs.shape[1]),
                style_imgs,
                torch.repeat_interleave(font_idx, trg_imgs.shape[1]),
                torch.tensor(trg_uni_ids),
                trg_imgs,
                content_imgs,
                trg_unis[0],
                torch.repeat_interleave(sample_index, style_imgs.shape[1]),
                sample_index,
                ref_unis[:self.k_shot]
            )
            # logging.debug(f"Returning dataset item: {ret}")
            return ret

    def __len__(self):
        length = sum([len(v) for v in self.avails.values()])
        # logging.debug(f"Dataset length: {length}")
        return length

    @staticmethod
    def collate_fn(batch):
        (style_ids, style_imgs, trg_ids, trg_uni_ids, trg_imgs, content_imgs, trg_unis, style_sample_index, trg_sample_index, ref_unis) = zip(*batch)
        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs, 1).unsqueeze_(2),
            torch.cat(trg_ids),
            torch.cat(trg_uni_ids),
            torch.cat(trg_imgs, 1).unsqueeze_(2),
            torch.cat(content_imgs, 1).unsqueeze_(2),
            trg_unis,
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            ref_unis
        )
        # logging.debug(f"Collate function output: {ret}")
        return ret


class CombTestDataset(Dataset):
    """
    CombTestDataset
    """
    def __init__(self, env, env_get, target_fu, avails, all_content_json, content_font, language="chn", transform=None, ret_targets=True):
        self.fonts = list(target_fu)
        self.n_uni_per_font = len(target_fu[list(target_fu)[0]])
        self.fus = [(fname, uni) for fname, unis in target_fu.items() for uni in unis]
        self.env = env
        self.env_get = env_get
        self.avails = avails
        self.transform = transform
        self.ret_targets = ret_targets
        self.content_font = content_font

        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16),
                       'kz_eng': lambda x: ord(x)
                       }

        self.to_int = to_int_dict[language.lower()]
        # logging.debug(f"Initialized CombTestDataset with {len(self.fonts)} fonts")

    def sample_pair_style(self, avail_unis):
        style_unis = random.sample(avail_unis, 3)
        # logging.debug(f"Sampled pair style unis: {style_unis}")
        return list(style_unis)

    def __getitem__(self, index):
        font_name, trg_uni = self.fus[index]
        font_idx = self.fonts.index(font_name)
        sample_index = torch.tensor([index])
        avail_unis = self.avails[font_name]
        style_unis = self.sample_pair_style(avail_unis)
        try:
            a = [self.env_get(self.env, font_name, uni, self.transform) for uni in style_unis]
        except Exception as e:
            logging.error(f"Error getting images for font {font_name} and unis {style_unis}: {e}")
        style_imgs = torch.stack(a)
        trg_dec_uni = torch.tensor([self.to_int(trg_uni)])
        content_img = self.env_get(self.env, self.content_font, trg_uni, self.transform)
        ret = (
            torch.repeat_interleave(torch.tensor([font_idx]), len(style_imgs)),
            style_imgs,
            torch.tensor([font_idx]),
            trg_dec_uni,
            torch.repeat_interleave(sample_index, len(style_imgs)),
            sample_index,
            content_img,
            trg_uni,
            style_unis
        )
        if self.ret_targets:
            try:
                trg_img = self.env_get(self.env, font_name, trg_uni, self.transform)
            except Exception as e:
                trg_img = torch.ones(size=(1, 128, 128))
                logging.error(f"Error getting target image for font {font_name} and uni {trg_uni}: {e}")
            ret += (trg_img,)
        # logging.debug(f"Returning dataset item: {ret}")
        return ret

    def __len__(self):
        length = len(self.fus)
        # logging.debug(f"Dataset length: {length}")
        return length

    @staticmethod
    def collate_fn(batch):
        (style_ids, style_imgs, trg_ids, trg_unis, style_sample_index, trg_sample_index, content_imgs, trg_uni, style_unis, *left) = list(zip(*batch))
        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs),
            torch.cat(trg_ids),
            torch.cat(trg_unis),
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            torch.cat(content_imgs).unsqueeze_(1),
            trg_uni,
            style_unis
        )
        if left:
            trg_imgs = left[0]
            ret += (torch.cat(trg_imgs).unsqueeze_(1),)
        # logging.debug(f"Collate function output: {ret}")
        return ret


class CombTrain_VQ_VAE_dataset(Dataset):
    """
    CombTrain_VQ_VAE_dataset,用于训练components码本的dataset，训练数据从content_font中取
    """

    def __init__(self, root, transform=None):
        self.img_path = root
        self.transform = transform
        self.imgs = self.read_file(self.img_path)
        # logging.debug(f"Initialized CombTrain_VQ_VAE_dataset with {len(self.imgs)} images")

    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        # logging.debug(f"Read files from path: {path}")
        return file_path_list

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img)
        # logging.debug(f"Returning image: {img_name}")
        return img

    def __len__(self):
        length = len(self.imgs)
        # logging.debug(f"Dataset length: {length}")
        return length


class FixedRefDataset(Dataset):
    '''
    FixedRefDataset
    '''
    def __init__(self, env, env_get, target_dict, ref_unis, k_shot,
                 all_content_json, content_font, language="chn", transform=None, ret_targets=True):
        self.target_dict = target_dict
        self.ref_unis = sorted(ref_unis)
        self.fus = [(fname, uni) for fname, unis in target_dict.items() for uni in unis]
        self.k_shot = k_shot
        with open(all_content_json, 'r') as f:
            self.cr_mapping = json.load(f)
        self.content_font = content_font
        self.fonts = list(target_dict)
        self.env = env
        self.env_get = env_get
        self.transform = transform
        self.ret_targets = ret_targets
        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                       }
        self.to_int = to_int_dict[language.lower()]
        # logging.debug(f"Initialized FixedRefDataset with {len(self.fus)} items")

    def sample_pair_style(self, font, style_uni):
        style_unis = random.sample(style_uni, 3)
        imgs = torch.cat([self.env_get(self.env, font, uni, self.transform) for uni in style_unis])
        # logging.debug(f"Sampled pair style images for font {font} with unis {style_unis}")
        return imgs, list(style_unis)

    def __getitem__(self, index):
        fname, trg_uni = self.fus[index]
        sample_index = torch.tensor([index])
        fidx = self.fonts.index(fname)
        avail_unis = list(set(self.ref_unis) - set([trg_uni]))
        style_imgs, style_unis = self.sample_pair_style(fname, self.ref_unis)
        fidces = torch.tensor([fidx])
        trg_dec_uni = torch.tensor([self.to_int(trg_uni)])
        style_dec_uni = torch.tensor([self.to_int(style_uni) for style_uni in style_unis])
        content_img = self.env_get(self.env, self.content_font, trg_uni, self.transform)
        ret = (
            torch.repeat_interleave(fidces, len(style_imgs)),
            style_imgs,
            fidces,
            trg_dec_uni,
            style_dec_uni,
            torch.repeat_interleave(sample_index, len(style_imgs)),
            sample_index,
            content_img,
            trg_uni,
            style_unis
        )
        if self.ret_targets:
            trg_img = self.env_get(self.env, fname, trg_uni, self.transform)
            ret += (trg_img,)
        # logging.debug(f"Returning dataset item: {ret}")
        return ret

    def __len__(self):
        length = len(self.fus)
        # logging.debug(f"Dataset length: {length}")
        return length

    @staticmethod
    def collate_fn(batch):
        style_ids, style_imgs, trg_ids, trg_unis, style_uni, style_sample_index, trg_sample_index, content_imgs, trg_uni, style_unis, *left = list(zip(*batch))
        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs).unsqueeze_(1),
            torch.cat(trg_ids),
            torch.cat(trg_unis),
            torch.cat(style_uni),
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            torch.cat(content_imgs).unsqueeze_(1),
            trg_uni,
            style_unis
        )
        if left:
            trg_imgs = left[0]
            ret += (torch.cat(trg_imgs).unsqueeze_(1),)
        # logging.debug(f"Collate function output: {ret}")
        return ret