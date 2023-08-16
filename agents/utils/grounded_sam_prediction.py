import argparse
import os
import gc
import sys
sys.path.append('./Grounded-Segment-Anything')
sys.path.append('./Grounded-Segment-Anything/GroundingDINO')

import numpy as np
import torch
from PIL import Image

import GroundingDINO.groundingdino.datasets.transforms as T
from segment_anything import build_sam, SamPredictor, build_sam_vit_l, build_sam_vit_b, sam_model_registry

from .grounding_dino_prediction import load_model, get_grounding_output
from constants import Starbucks_ROOM, TG_ROOM, NursingRoom_ROOM


class SemanticPredGroundedSAM():

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

        # initialize SAM
        self.predictor = SamPredictor(sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint).to(self.device))
        torch.cuda.set_device(args.sem_gpu_id)

    def get_prediction(self, img, return_obj=False, detect_room=False):
        gc.collect()

        H, W, _ = img.shape
        # convert to RGB
        rgb = img[:, :, [2, 1, 0]]
        image = Image.fromarray(rgb.astype(np.uint8))
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
        if boxes_filt.shape[0] > 0:
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            self.predictor.set_image(rgb)
            boxes_filt = boxes_filt.cpu()
            transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, (H, W)).to(self.device)

            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.device),
                multimask_output=False,
            )

            masks = masks.cpu().numpy() * 1.0

            for j in range(len(pred_ids)):
                class_idx = pred_ids[j]
                obj_mask = masks[j]
                semantic_input[:, :, class_idx] += np.squeeze(obj_mask)

                if return_obj:
                    score = logits_filt[j]
                    box = boxes_filt[j].cpu().numpy().astype(int)
                    center_point = (box[2:] - box[:2]) // 2
                    # label, score, cx, cy
                    detect_objects.append([class_idx, score, center_point[0], center_point[1]])

        if return_obj:
            return semantic_input, img, detect_objects
        else:
            return semantic_input, img

