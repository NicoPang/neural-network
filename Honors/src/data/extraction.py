import os
from PIL import Image

# General extraction function
def extract_features_from_image(path):
    im = Image.open(path, 'r')
    pixel_values = list(im.getdata())
    # Invert colors - not sure if necessary, but should be harmless at worst
    alphas = list(map(lambda x: 1 - x/255, pixel_values))
    return alphas

def get_choice_classes():
    numbers = [chr(i + 48) for i in range(10)]
    lowers = [chr(i + 97) for i in range(26)]
    uppers = [chr(i + 65) for i in range(26)]
    return numbers + lowers + uppers

# Returns all of the image data from the CHoiCe dataset
def extract_choice_dataset():
    all_classes = get_choice_classes()

    vectors = {}

    for c in all_classes:
        vectors[c] = []

    for i in range(len(all_classes)):
        path_to_folder = "../Datasets/CHoiCe/V0.3/data" + "/{}/".format(i)
        files = os.listdir(path_to_folder)
        for file in files:
            feature_vector = extract_features_from_image(path_to_folder + file)
            vectors[all_classes[i]].append(feature_vector)

    return vectors