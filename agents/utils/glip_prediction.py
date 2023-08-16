import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

from constants import Starbucks_ROOM, TG_ROOM, NursingRoom_ROOM


class SemanticPredGLIP():

    def __init__(self, args, categories=None):
        if categories is not None:
            self.caption = ' . '.join(categories) + '.'
        else:
            self.caption = None
        self.num_sem_categories = args.num_sem_categories

        if args.split == 'Starbucks':
            self.room_captions = ' . '.join(Starbucks_ROOM) + '.'
            self.num_room_categories = len(Starbucks_ROOM) + 1
        elif args.split == 'TG':
            self.room_captions = ' . '.join(TG_ROOM) + '.'
            self.num_room_categories = len(TG_ROOM) + 1
        elif args.split == 'NursingRoom':
            self.room_captions = ' . '.join(NursingRoom_ROOM) + '.'
            self.num_room_categories = len(NursingRoom_ROOM) + 1

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

    def get_prediction(self, img, return_obj=False, detect_room=False, caption=None):
        h, w, _ = img.shape
        if detect_room:
            top_predictions = self.glip_demo.inference(img, self.room_captions)
            num_cats = self.num_room_categories
        elif caption is not None:
            top_predictions = self.glip_demo.inference(img, caption)
            num_cats = self.num_sem_categories
        else:
            top_predictions = self.glip_demo.inference(img, self.caption)
            num_cats = self.num_sem_categories
        # print(top_predictions)

        semantic_input = np.zeros((h, w, num_cats))

        labels = top_predictions.get_field('labels').numpy()
        scores = top_predictions.get_field('scores')
        print('seg labels: ', labels)
        print('seg scores: ', scores)
        _, idx = scores.sort(0)
        if self.glip_demo.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0

        detect_objects = []
        for j in idx.tolist():
            class_idx = labels[j]
            if class_idx <= num_cats:
                obj_bbox = top_predictions.bbox[j].to(torch.int64).numpy()
                score = scores.numpy()[j]
                semantic_input[obj_bbox[1]:obj_bbox[3], obj_bbox[0]:obj_bbox[2], class_idx - plus] = score

                if return_obj:
                    center_point = (obj_bbox[2:] - obj_bbox[:2]) // 2
                    # label, score, cx, cy
                    detect_objects.append([class_idx, score, center_point[0], center_point[1]])

        if return_obj:
            return semantic_input, img, detect_objects
        else:
            return semantic_input, img

