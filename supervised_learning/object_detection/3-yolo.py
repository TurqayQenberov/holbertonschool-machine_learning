#!/usr/bin/env python3
"""Defines the Yolo class for performing object detection using YOLOv3"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path: path to Darknet Keras model
        classes_path: path to list of class names used for the Darknet model
        class_t: box score threshold for the initial filtering step
        nms_t: IOU threshold for non-max suppression
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Applies the sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        outputs: list of numpy.ndarrays containing the predictions from the
            Darknet model for a single image
        image_size: numpy.ndarray containing the image's original size
            [image_height, image_width]

        Returns: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_prob = self.sigmoid(output[..., 5:])

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

            # Center coordinates
            b_xy = self.sigmoid(t_xy)
            grid = np.tile(
                np.indices((grid_width, grid_height)).T,
                anchor_boxes
            ).reshape((grid_height, grid_width, anchor_boxes, 2))
            b_xy = (b_xy + grid) / [grid_width, grid_height]

            # Width and height
            anchors = self.anchors[i]
            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]
            b_wh = (np.exp(t_wh) * anchors) / [input_width, input_height]

            # Corner coordinates
            x1y1 = b_xy - (b_wh / 2)
            x2y2 = b_xy + (b_wh / 2)

            box = np.concatenate((x1y1, x2y2), axis=-1)

            # Scale to original image size
            box[..., 0] *= image_width
            box[..., 1] *= image_height
            box[..., 2] *= image_width
            box[..., 3] *= image_height

            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        boxes: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 4)
        box_confidences: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1)
        box_class_probs: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes)

        Returns: (filtered_boxes, box_classes, box_scores)
        """
        boxes_list = []
        box_classes_list = []
        box_scores_list = []

        for i in range(len(boxes)):
            box_scores_i = box_confidences[i] * box_class_probs[i]
            box_classes_i = np.argmax(box_scores_i, axis=-1)
            box_class_scores_i = np.max(box_scores_i, axis=-1)

            mask = box_class_scores_i >= self.class_t

            boxes_list.append(boxes[i][mask])
            box_classes_list.append(box_classes_i[mask])
            box_scores_list.append(box_class_scores_i[mask])

        filtered_boxes = np.concatenate(boxes_list, axis=0)
        box_classes = np.concatenate(box_classes_list, axis=0)
        box_scores = np.concatenate(box_scores_list, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        filtered_boxes: numpy.ndarray of shape (?, 4) containing all of the
            filtered bounding boxes
        box_classes: numpy.ndarray of shape (?,) containing the class
            number for the class that filtered_boxes predicts, respectively
        box_scores: numpy.ndarray of shape (?) containing the box scores
            for each box in filtered_boxes, respectively

        Returns: (box_predictions, predicted_box_classes,
            predicted_box_scores)
        """
        unique_classes = np.unique(box_classes)

        box_predictions_list = []
        predicted_box_classes_list = []
        predicted_box_scores_list = []

        for cls in unique_classes:
            cls_mask = box_classes == cls

            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]

            areas = (x2 - x1) * (y2 - y1)

            # Sort indices by score descending
            order = np.argsort(cls_scores)[::-1]

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
                intersection = w * h

                union = areas[i] + areas[order[1:]] - intersection
                iou = intersection / union

                remaining = np.where(iou <= self.nms_t)[0]
                order = order[remaining + 1]

            keep = np.array(keep)

            box_predictions_list.append(cls_boxes[keep])
            predicted_box_classes_list.append(
                np.full(keep.shape[0], cls, dtype=box_classes.dtype)
            )
            predicted_box_scores_list.append(cls_scores[keep])

        box_predictions = np.concatenate(box_predictions_list, axis=0)
        predicted_box_classes = np.concatenate(
            predicted_box_classes_list, axis=0
        )
        predicted_box_scores = np.concatenate(
            predicted_box_scores_list, axis=0
        )

        return box_predictions, predicted_box_classes, predicted_box_scores
