#!/usr/bin/env python3
"""Yolo class - preprocess images"""
import numpy as np
import tensorflow.keras as K
import glob
import cv2
import os


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process Darknet outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            box = output[..., :4]
            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_prob = self.sigmoid(output[..., 5:])

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

            tx = box[..., 0]
            ty = box[..., 1]
            tw = box[..., 2]
            th = box[..., 3]

            cx = np.tile(np.arange(grid_w), grid_h).reshape(grid_h, grid_w)
            cy = np.tile(np.arange(grid_h), grid_w)
            cy = cy.reshape(grid_w, grid_h).T

            cx = np.tile(cx[..., np.newaxis], (1, 1, anchor_boxes))
            cy = np.tile(cy[..., np.newaxis], (1, 1, anchor_boxes))

            bx = (self.sigmoid(tx) + cx) / grid_w
            by = (self.sigmoid(ty) + cy) / grid_h

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]

            bw = (np.exp(tw) * anchor_w) / input_w
            bh = (np.exp(th) * anchor_h) / input_h

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes based on class threshold"""
        box_scores_full = []
        for bc, bcp in zip(box_confidences, box_class_probs):
            box_scores_full.append(bc * bcp)

        box_scores_list = [
            bs.reshape(-1, bs.shape[-1]) for bs in box_scores_full
        ]
        box_scores_concat = np.concatenate(box_scores_list, axis=0)

        box_classes = np.argmax(box_scores_concat, axis=-1)
        box_class_scores = np.max(box_scores_concat, axis=-1)

        filter_mask = box_class_scores >= self.class_t

        boxes_list = [b.reshape(-1, 4) for b in boxes]
        boxes_concat = np.concatenate(boxes_list, axis=0)

        filtered_boxes = boxes_concat[filter_mask]
        box_classes = box_classes[filter_mask]
        box_scores = box_class_scores[filter_mask]

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)

            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]

            areas = (x2 - x1) * (y2 - y1)
            order = cls_scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                inter = w * h

                iou = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

            keep = np.array(keep)

            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(
                np.full(keep.shape[0], cls, dtype=np.int32)
            )
            predicted_box_scores.append(cls_scores[keep])

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(
            predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(
            predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Load images from a folder"""
        image_paths = glob.glob(folder_path + '/*')
        images = [cv2.imread(path) for path in image_paths]
        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocess images for the Darknet model

        images: list of numpy.ndarray images

        Returns:
            pimages: numpy.ndarray of shape (ni, input_h, input_w, 3)
            image_shapes: numpy.ndarray of shape (ni, 2)
        """
        input_h = self.model.input.shape[2]
        input_w = self.model.input.shape[1]

        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append(img.shape[:2])

            resized = cv2.resize(
                img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            rescaled = resized / 255.0
            pimages.append(rescaled)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
