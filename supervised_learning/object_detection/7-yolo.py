def predict(self, folder_path):
        """
        folder_path: a string representing the path to the folder holding
            all the images to predict

        Returns: (predictions, image_paths)
            predictions: a list of tuples for each image of
                (boxes, box_classes, box_scores)
            image_paths: a list of image paths corresponding to each
                prediction in predictions
        """
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
