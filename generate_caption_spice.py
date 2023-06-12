local_path = 'refcocog/images/'
local_annotations = 'refcocog/annotations/'

import json
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

import stanza
from tqdm import tqdm

from transformers import ViTFeatureExtractor, VisionEncoderDecoderModel
from transformers import RobertaTokenizerFast

#ignore warnings
import warnings
warnings.filterwarnings("ignore")


def clear_caption(caption):
    caption = caption.replace('<s>', '')
    caption = caption.replace('</s>', '')
    return caption


# remove the id in the image name string
def split_string(string):
    string = string.split("_")
    string = string[:-1]
    string = "_".join(string)
    append = ".jpg"
    string = string + append

    return string

# dataset class definition
class Coco(Dataset):
    def __init__(self, path_json, path_pickle, train=True):
        self.path_json = path_json
        self.path_pickle = path_pickle
        self.train = train

        # load images and annotations
        with open(self.path_json) as json_data:
            data = json.load(json_data)
            self.ann_frame = pd.DataFrame(data['annotations'])
            self.ann_frame = self.ann_frame.reset_index(drop=False)

        with open(self.path_pickle, 'rb') as pickle_data:
            data = pickle.load(pickle_data)
            self.refs_frame = pd.DataFrame(data)

        # separate each sentence in dataframe
        self.refs_frame = self.refs_frame.explode('sentences')
        self.refs_frame = self.refs_frame.reset_index(drop=False)

        self.size = self.refs_frame.shape[0]

        # merge the dataframes
        self.dataset = pd.merge(
            self.refs_frame, self.ann_frame, left_on='ann_id', right_on='id')
        # drop useless columns for cleaner and smaller dataset
        self.dataset = self.dataset.drop(columns=['segmentation', 'id', 'category_id_y', 'ref_id', 'index_x',
                                         'iscrowd', 'image_id_y', 'image_id_x', 'category_id_x', 'ann_id', 'sent_ids', 'index_y', 'area'])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset.iloc[idx]

    def get_annotation(self, idx):
        return self.ann_frame.iloc[idx]

    def get_imgframe(self, idx):
        return self.img_frame.iloc[idx]

    def get_validation(self):
        return self.dataset[self.dataset['split'] == 'val']

    def get_test(self):
        return self.dataset[self.dataset['split'] == 'test']

    def get_train(self):
        return self.dataset[self.dataset['split'] == 'train']
    

# dataset load
dataset = Coco(local_annotations + 'instances.json', local_annotations + "refs(umd).p")

dataframe = dataset.get_validation()
#define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length=20)
text_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
text_model = VisionEncoderDecoderModel.from_pretrained('/home/pappol/Scrivania/deepLearning/Image_Captioning_VIT_Roberta_final_4')
text_model.to(device)


with open('image_captioning_spice.json', 'w') as f:
    entries = []
    for i in tqdm(range(len(dataframe))):
        input = dataframe.iloc[i]
        image_path = split_string(input["file_name"])
        sentence = input["sentences"]["raw"]
        original_img = Image.open(local_path + image_path).convert("RGB")
        #crop image based on ground truth
        bbox = input["bbox"]
        cropped_img = original_img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        features = text_feature_extractor(cropped_img, return_tensors="pt").pixel_values.to(device)
        generated = text_model.generate(features)[0].to(device)
        caption = text_tokenizer.decode(generated)
        caption = clear_caption(caption)

        entry = {'sentence': caption,
                'index': i}
        entries.append(entry)
        
    json.dump(entries, f, indent=4)
