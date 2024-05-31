import numpy as np

import support.processing as processing

class Classifier:
    def __init__(self, model):
        self.model = model
        
        self.training_dataset = None
        self.testing_dataset = None

    def create_training_dataset(self, dataset):
        dataset.shuffle()
        self.training_dataset = dataset

    def create_testing_dataset(self, dataset):
        dataset.shuffle()
        self.testing_dataset = dataset

    def initialize_model(self, f = processing.xavier_initialize):
        processing.initialize(f, self.model)

    def get_changes(self, data):
            output = processing.forward_propogate(self.model, data.get_feature_vector())
            expected = self.training_dataset.get_expected_vector(data.get_class_name())
            # print("expected")
            # print(expected)
            changes = processing.back_propogate(self.model, output, expected)
            return changes
    
    def train(self):
        for data in self.training_dataset.get_dataset():
            changes = self.get_changes(data)
            # print("changes")
            # print(changes)
            self.model.adjust_weights_biases(changes[0], changes[1])

    # 
    def segmented_train(self, n):
        rem = self.training_dataset.get_dataset_size()

        if n > rem:
            print("ERROR: More segments than data.")
            return

        batch_size = rem//n

        data_index = 0
        while rem > 0:
            weight_changes = self.model.generate_blank_weights()
            bias_changes = self.model.generate_blank_biases()

            for i in range(batch_size):
                if rem == 0:
                    break

                changes = self.get_changes()
                weight_changes += changes[0]
                bias_changes += changes[1]

                data_index += 1
                rem -= 1

            self.model.adjust_weights_biases(weight_changes, bias_changes)

    def classify(self, feature_vector):
        output = processing.forward_propogate(self.model, feature_vector)
        output_vector = []
        for num in output[-1]:
            output_vector.append(self.model.activate(num))
        return output_vector

    def test_classified(self):
        classified_data = []

        for data in self.testing_dataset.get_dataset():
            classified_data.append((data, self.classify(data.get_feature_vector())))

        return classified_data

    def print_dataset_info(self):
        print("TRAINING: {}: {} entries".format(self.training_dataset.get_name(), self.training_dataset.get_dataset_size()))
        print("TESTING: {}: {} entries".format(self.testing_dataset.get_name(), self.testing_dataset.get_dataset_size()))

    def print_classified_data_raw(self, classified_data):
        for entry in classified_data:
            # rounded = [round(i, 3) for i in entry[1]]
            print("picture of {}: {}".format(entry[0].get_class_name(), entry[1]))

    def print_classified_data(self, classified_data):
        for entry in classified_data:
            normalized = entry[1]/np.sum(entry[1])
            rounded = [round(i, 3) for i in normalized]
            print("picture of {}: {}".format(entry[0].get_class_name(), rounded))

    def print_classified_data_stats(self, classified_data):
        classes = self.training_dataset.get_classes()
        num_outputs = self.training_dataset.get_number_outputs()

        total_entries = [0 for _ in range(num_outputs)]
        total_correct = [0 for _ in range(num_outputs)]
        total_confidences = [[0 for _ in range(num_outputs)] for _ in range(num_outputs)]
        total_classified = [[0 for _ in range(num_outputs)] for _ in range(num_outputs)]

        for entry in classified_data:
            data = entry[0]
            output = entry[1]
            normalized = output/np.sum(output)
            rounded = [round(i, 3) for i in normalized]
            class_index = np.argmax(self.training_dataset.get_expected_vector(data.get_class_name()))
            total_entries[class_index] += 1
            for i in range(num_outputs):
                total_confidences[class_index][i] += rounded[i]
            total_classified[class_index][np.argmax(rounded)] += 1
            if class_index == np.argmax(rounded):
                total_correct[class_index] += 1

        for i in range(num_outputs):
            print("Class \"{}\": accuracy = {}, confidence = {}, classified as = {} at a probability of {}".format(
                classes[i],
                round(total_correct[i] / total_entries[i], 3),
                round(total_confidences[i][i] / total_entries[i], 3),
                classes[np.argmax(total_confidences[i])],
                round(total_classified[i][np.argmax(total_confidences[i])] / total_entries[i], 3)
            ))