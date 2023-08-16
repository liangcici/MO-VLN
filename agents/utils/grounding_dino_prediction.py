import argparse
import os
import sys
sys.path.append('./Grounded-Segment-Anything')
sys.path.append('./Grounded-Segment-Anything/GroundingDINO')

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from constants import Starbucks_ROOM, TG_ROOM, NursingRoom_ROOM


def load_model(model_config_path, model_checkpoint_path, device="cpu", device_id=0):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.device_id = device_id
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, categories, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_ids = []
    logits = []
    for logit, box in zip(logits_filt, boxes_filt):
        sort_indexes = logit.argsort(descending=True)
        index = 0
        token_id = tokenized["input_ids"][sort_indexes[index]]
        pred_word = tokenlizer.decode(token_id)
        while '#' in pred_word:
            index += 1
            token_id = tokenized["input_ids"][sort_indexes[index]]
            pred_word = tokenlizer.decode(token_id)
        # print(token_id, pred_word)
        for cat in categories:
            if pred_word in cat:
                pred_id = categories[cat]
                break
        pred_ids.append(pred_id)
        logits.append(logit[sort_indexes[index]].cpu().numpy())

    return boxes_filt, pred_ids, logits


class SemanticPredGroundingDINO():

    def __init__(self, args, categories):
        self.categories = {}
        for id, cat in enumerate(categories):
            self.categories[cat] = id
        self.caption = ' . '.join(categories) + '.'

        self.num_sem_categories = args.num_sem_categories

        self.room_categories = {}
        if args.split == 'Starbucks':
            self.room_captions = ' . '.join(Starbucks_ROOM) + '.'
            self.num_room_categories = len(Starbucks_ROOM) + 1
            for id, cat in enumerate(Starbucks_ROOM):
                self.room_categories[cat] = id
        elif args.split == 'TG':
            self.room_captions = ' . '.join(TG_ROOM) + '.'
            self.num_room_categories = len(TG_ROOM) + 1
            for id, cat in enumerate(TG_ROOM):
                self.room_categories[cat] = id
        elif args.split == 'NursingRoom':
            self.room_captions = ' . '.join(NursingRoom_ROOM) + '.'
            self.num_room_categories = len(NursingRoom_ROOM) + 1
            for id, cat in enumerate(NursingRoom_ROOM):
                self.room_categories[cat] = id

        # cfg
        config_file = args.det_config_file  # change the path of the model config file
        grounded_checkpoint = args.det_weight  # change the path of the model
        self.box_threshold = args.det_thresh
        self.text_threshold = args.det_thresh
        self.device = "cuda:0"

        # load model
        self.model = load_model(config_file, grounded_checkpoint, device=self.device, device_id=args.sem_gpu_id)

        self.transform = T.Compose(
                            [
                                # T.RandomResize([800], max_size=1333),
                                T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            ]
                        )
        torch.cuda.set_device(args.sem_gpu_id)

    def get_prediction(self, img, return_obj=False, detect_room=False):
        H, W, _ = img.shape
        # convert to RGB
        image = img[:, :, [2, 1, 0]]
        image = Image.fromarray(image.astype(np.uint8))
        image, _ = self.transform(image, None)  # 3, h, w

        if detect_room:
            caption = self.room_captions
            num_cats = self.num_room_categories
            categories = self.room_categories
        else:
            caption = self.caption
            num_cats = self.num_sem_categories
            categories = self.categories

        # run model
        boxes_filt, pred_ids, logits_filt = get_grounding_output(
            self.model, image, caption, self.box_threshold, self.text_threshold, categories, device=self.device
        )
        print(pred_ids)

        semantic_input = np.zeros((H, W, num_cats))

        detect_objects = []
        for j in range(len(pred_ids)):
            class_idx = pred_ids[j]
            # from 0..1 to 0..W, 0..H
            box = boxes_filt[j] * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            box = box.cpu().numpy().astype(int)
            semantic_input[box[1]:box[3], box[0]:box[2], class_idx] = 1.

            if return_obj:
                score = logits_filt[j]
                center_point = (box[2:] - box[:2]) // 2
                # label, score, cx, cy
                detect_objects.append([class_idx, score, center_point[0], center_point[1]])

        if return_obj:
            return semantic_input, img, detect_objects
        else:
            return semantic_input, img

