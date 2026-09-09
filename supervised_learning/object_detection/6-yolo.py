cat > 6-yolo.py << 'PYEOF'
#!/usr/bin/env python3
"""Defines the Yolo class for performing object detection using YOLOv3"""
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

    def sigmoid(self, x):
        """Applies the sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process Darknet outputs into boxes, confidences, class probs"""
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

            b_xy = self.sigmoid(t_xy)
            grid = np.tile(
                np.indices((grid_width, grid_height)).T,
                anchor_boxes
            ).reshape((grid_height, grid_width, anchor_boxes, 2))
            b_xy = (b_xy + grid) / [grid_width, grid_height]

            anchors = self.anchors[i]
            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]
            b_wh = (np.exp(t_wh) * anchors) / [input_width, input_height]

            x1y1 = b_xy - (b_wh / 2)
            x2y2 = b_xy + (b_wh / 2)

            box = np.concatenate((x1y1, x2y2), axis=-1)

            box[..., 0] *= image_width
            box[..., 1] *= image_height
            box[..., 2] *= image_width
            box[..., 3] *= image_height

            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes below the class threshold"""
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
        """Apply non-max suppression per class"""
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

    @staticmethod
    def load_images(folder_path):
        """Load all images from a folder"""
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')

        image_paths = sorted(
            p for p in glob.glob(os.path.join(folder_path, '*'))
            if os.path.isfile(p) and p.lower().endswith(valid_ext)
        )
        images = [cv2.imread(image_path) for image_path in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """Resize and rescale images for the Darknet model"""
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages_list = []
        image_shapes_list = []

        for img in images:
            image_shapes_list.append(img.shape[:2])

            resized = cv2.resize(
                img, (input_w, input_h), interpolation=cv2.INTER_CUBIC
            )
            resized = resized.astype(np.float32) / 255.0

            pimages_list.append(resized)

        pimages = np.array(pimages_list)
        image_shapes = np.array(image_shapes_list)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Draw boxes, class names, and scores on an image and display it"""
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            class_name = self.class_names[box_classes[i]]
            score = str(round(box_scores[i], 2))
            text = "{} {}".format(class_name, score)

            cv2.putText(
                image,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            cv2.imwrite(os.path.join('detections', file_name), image)

        cv2.destroyAllWindows()
PYEOF
cat > 7-yolo.py << 'PYEOF'
#!/usr/bin/env python3
"""Defines the Yolo class for performing object detection using YOLOv3"""
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

    def sigmoid(self, x):
        """Applies the sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process Darknet outputs into boxes, confidences, class probs"""
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

            b_xy = self.sigmoid(t_xy)
            grid = np.tile(
                np.indices((grid_width, grid_height)).T,
                anchor_boxes
            ).reshape((grid_height, grid_width, anchor_boxes, 2))
            b_xy = (b_xy + grid) / [grid_width, grid_height]

            anchors = self.anchors[i]
            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]
            b_wh = (np.exp(t_wh) * anchors) / [input_width, input_height]

            x1y1 = b_xy - (b_wh / 2)
            x2y2 = b_xy + (b_wh / 2)

            box = np.concatenate((x1y1, x2y2), axis=-1)

            box[..., 0] *= image_width
            box[..., 1] *= image_height
            box[..., 2] *= image_width
            box[..., 3] *= image_height

            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes below the class threshold"""
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
        """Apply non-max suppression per class"""
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

    @staticmethod
    def load_images(folder_path):
        """Load all images from a folder"""
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')

        image_paths = sorted(
            p for p in glob.glob(os.path.join(folder_path, '*'))
            if os.path.isfile(p) and p.lower().endswith(valid_ext)
        )
        images = [cv2.imread(image_path) for image_path in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """Resize and rescale images for the Darknet model"""
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages_list = []
        image_shapes_list = []

        for img in images:
            image_shapes_list.append(img.shape[:2])

            resized = cv2.resize(
                img, (input_w, input_h), interpolation=cv2.INTER_CUBIC
            )
            resized = resized.astype(np.float32) / 255.0

            pimages_list.append(resized)

        pimages = np.array(pimages_list)
        image_shapes = np.array(image_shapes_list)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Draw boxes, class names, and scores on an image and display it"""
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            class_name = self.class_names[box_classes[i]]
            score = str(round(box_scores[i], 2))
            text = "{} {}".format(class_name, score)

            cv2.putText(
                image,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            cv2.imwrite(os.path.join('detections', file_name), image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """Run full prediction pipeline on all images in a folder"""
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        outputs = self.model.predict(pimages)

        predictions = []

        for i in range(len(images)):
            output_i = [output[i] for output in outputs]

            boxes, box_confidences, box_class_probs = self.process_outputs(
                output_i, image_shapes[i]
            )

            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs
            )

            box_predictions, predicted_box_classes, predicted_box_scores = \
                self.non_max_suppression(
                    filtered_boxes, box_classes, box_scores
                )

            predictions.append(
                (box_predictions, predicted_box_classes,
                 predicted_box_scores)
            )

            file_name = os.path.basename(image_paths[i])
            self.show_boxes(
                images[i], box_predictions, predicted_box_classes,
                predicted_box_scores, file_name
            )

        return predictions, image_paths
PYEOF
