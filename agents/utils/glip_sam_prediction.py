import argparse
import os
import gc
import sys
sys.path.append('./Grounded-Segment-Anything')

import numpy as np
import torch
from PIL import Image

from segment_anything import build_sam, SamPredictor, build_sam_vit_l, build_sam_vit_b, sam_model_registry
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo


class SemanticPredGLIPSAM():

    def __init__(self, args, categories):
        self.caption = ' . '.join(categories) + '.'
        self.num_sem_categories = args.num_sem_categories

        # update the config options with the config file
        # manual override some options
        cfg.local_rank = 0
        cfg.num_gpus = 1
        cfg.merge_from_file(args.det_config_file)
        cfg.merge_from_list(["MODEL.WEIGHT", args.det_weight])
        cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

        self.glip_demo = GLIPDemo(
            cfg,
            # min_image_size=800,
            confidence_threshold=args.det_thresh,
            show_mask_heatmaps=False
        )
        self.thresh = args.det_thresh
        self.device = "cuda:0"

        # initialize SAM
        self.predictor = SamPredictor(sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint).to(self.device))
        torch.cuda.set_device(args.sem_gpu_id)

    def get_prediction(self, img):
        H, W, _ = img.shape
        # convert to RGB
        rgb = img[:, :, [2, 1, 0]]

        top_predictions = self.glip_demo.inference(img, self.caption)
        print(top_predictions)

        labels = top_predictions.get_field('labels').numpy()
        print('seg labels: ', labels)

        if self.glip_demo.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0

        semantic_input = np.zeros((H, W, self.num_sem_categories))

        if labels.shape[0] > 0:
            boxes_filt = top_predictions.bbox

            self.predictor.set_image(rgb)
            transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, (H, W)).to(self.device)

            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.device),
                multimask_output=False,
            )

            masks = masks.cpu().numpy() * 1.0

            for j in range(len(labels)):
                class_idx = labels[j]
                if class_idx <= self.num_sem_categories:
                    obj_mask = masks[j]
                    semantic_input[:, :, class_idx - plus] += np.squeeze(obj_mask)

        return semantic_input, img
