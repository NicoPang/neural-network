import random

class UnclassifiedData:
    def __init__(self, name, feature_vector):
        self.name = name
        self.feature_vector = feature_vector

    def get_name(self):
        return self.name

    def get_feature_vector(self):
        return self.feature_vector

    def get_number_features(self):
        return len(self.feature_vector)

    def __str__(self):
        return "{}: {}".format(self.name, self.feature_vector)

class ClassifiedData(UnclassifiedData):
    def __init__(self, name, feature_vector, class_name = None):
        super().__init__(name, feature_vector)
        self.class_name = class_name
    
    def get_class_name(self):
        return self.class_name

    def __str__(self):
        return "Feature vector: {}, class name: {}".format(self.feature_vector, self.class_name)

class UnclassifiedDataset:
    def __init__(self, name):
        self.name = name
        self.number_inputs = None

        self.dataset = []

    def get_name(self):
        return self.name

    def get_number_inputs(self):
        return self.number_inputs

    def get_data(self, index):
        return self.dataset[index]

    def get_dataset(self):
        return self.dataset
        
    def get_dataset_size(self):
        return len(self.dataset)

    def add_dataset(self, list_data):
        for data in list_data:
            self.add_data(data)

    def add_data(self, data):

        if self.number_inputs == None:
            self.number_inputs = data.get_number_features()
        elif self.number_inputs != data.get_number_features():
            print("Error: unable to add data '{}' due to mismatched feature size ({} != {}).".format(data.get_name(), data.get_number_features(), self.number_inputs))
            return

        self.dataset.append(data)

    def shuffle(self):
        random.shuffle(self.dataset)


class ClassifiedDataset(UnclassifiedDataset):
    def __init__(self, name, classes):
        super().__init__(name)

        self.number_outputs = len(classes)
        self.classes = classes

    def add_data(self, data):
        if not self.is_valid_class(data.get_class_name()):
            print("Error: unable to add data '{}' due to nonexistent class '{}'.".format(data.get_name(), data.get_class_name()))
            return

        super().add_data(data)

    def get_number_outputs(self):
        return self.number_outputs

    def get_classes(self):
        return self.classes

    def is_valid_class(self, id):
        for c in self.classes:
            if c == id:
                return True
        
        return False

    def get_expected_vector(self, class_name):
        exp_out = []

        for c_name in self.classes:
            exp_out.append(1 if c_name == class_name else 0)

        return exp_out