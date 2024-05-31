import sys

from data.dataset import ClassifiedData, ClassifiedDataset
from data.extraction import get_choice_classes, extract_choice_dataset
from ml.classifier import Classifier
from ml.network import BasicNeuralNetworkModel

CHOICE_ALL = get_choice_classes()
CHOICE_NUMS = [chr(i + 48) for i in range(10)]
CHOICE_ALPHA = [chr(i + 97) for i in range(26)] + [chr(i + 65) for i in range(26)]
CHOICE_UPPER = [chr(i + 65) for i in range(26)]
CHOICE_LOWER = [chr(i + 97) for i in range(26)]

# CHANGE IF YOU WANT CUSTOM HIDDEN LAYERS
HIDDEN_LAYERS = [85]

def build_classifier(chars):
    choice_dataset = extract_choice_dataset()

    training_dataset = ClassifiedDataset("training-dataset", chars)
    testing_dataset = ClassifiedDataset("testing-dataset", chars)

    for char in chars:
        num_entries = len(choice_dataset[char])

        # Add training data
        counter = 0
        for data in choice_dataset[char][:30]:
            training_dataset.add_data(ClassifiedData(counter, data, char))

        # Add testing data
        couter = 0
        for data in choice_dataset[char][30:40]:
            testing_dataset.add_data(ClassifiedData(couter, data, char))

    layers = [training_dataset.get_number_inputs()]

    for i in HIDDEN_LAYERS:
        layers.append(i)
    
    layers.append(training_dataset.get_number_outputs())
    model = BasicNeuralNetworkModel(layers)
    classifier = Classifier(model)
    classifier.initialize_model()
    
    classifier.create_training_dataset(training_dataset)
    classifier.create_testing_dataset(testing_dataset)

    return classifier

if __name__ == '__main__':

    # Parse arguments
    if len(sys.argv) == 1:
        print("usage: python3 run_network.py args...")
        print("args is either a list of characters to compare or one of the following keywords:")
        print("all   -> compares all characters")
        print("num   -> compares all numbers")
        print("alpha -> compares all letters")
        print("upper -> compares only uppercase letters")
        print("lower -> compares only lowercase letters")
        sys.exit()

    classifier = None

    # Build classifier
    if sys.argv[1] == "all":
        classifier = build_classifier(CHOICE_ALL)
    elif sys.argv[1] == "num":
        classifier = build_classifier(CHOICE_NUMS)
    elif sys.argv[1] == "alpha":
        classifier = build_classifier(CHOICE_ALPHA)
    elif sys.argv[1] == "upper":
        classifier = build_classifier(CHOICE_UPPER)
    elif sys.argv[1] == "lower":
        classifier = build_classifier(CHOICE_LOWER)
    else:
        chars = sys.argv[1:]

        choice_classes = get_choice_classes()
        for char in chars:
            if char not in choice_classes:
                print("error: bad arg - " + char)
                sys.exit()
        
        classifier = build_classifier(chars)

    classifier.print_dataset_info()
    classifier.train()
    classified_data = classifier.test_classified()
    # classifier.print_classified_data(classified_data)
    classifier.print_classified_data_stats(classified_data)