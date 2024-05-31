from data.dataset import ClassifiedData, ClassifiedDataset
from ml.classifier import Classifier
from ml.network import BasicNeuralNetworkModel
import support.processing as processing

if __name__ == '__main__':
    dataset = ClassifiedDataset("test-dataset", ["zero", "one"])

    test_set = ClassifiedDataset("tt ", ["zero", "one"])
    
    for i in range(100):
        dataset.add_data(ClassifiedData("zero {}".format(i), [0, 0, 0, 0, 0], "zero"))
        test_set.add_data(ClassifiedData("zero {}".format(i), [0, 0, 0, 0, 0], "zero"))

    for i in range(100):
        dataset.add_data(ClassifiedData("one {}".format(i), [1, 1, 1, 1, 1], "one"))
        test_set.add_data(ClassifiedData("one {}".format(i), [1, 1, 1, 1, 1], "one"))

    dataset.shuffle()

    model = BasicNeuralNetworkModel([5, 2])
    processing.initialize(processing.xavier_initialize, model)
    classifier = Classifier(model)
    classifier.create_training_dataset(dataset)
    classifier.create_testing_dataset(test_set)
    classifier.train()
    print(classifier.classify([1, 1, 1, 1, 1]))
    classified_data = classifier.test_classified()
    classifier.print_classified_data_raw(classified_data)