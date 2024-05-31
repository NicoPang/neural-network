from data.dataset import ClassifiedData, ClassifiedDataset

if __name__ == '__main__':
    d1 = ClassifiedData("test01", [1, 1], "test")
    d2 = ClassifiedData("test02", [0, 1], "test")
    d3 = ClassifiedData("test03", [1, 0], "test")

    d21 = ClassifiedData("testb01", [1, 1], "other")
    d22 = ClassifiedData("testb02", [1, 1], "other")
    d23 = ClassifiedData("testb03", [0, 0], "other")
    
    dlarge = ClassifiedData("too-big", [1, 1, 1], "test")
    dfake = ClassifiedData("not-real", [1, 1], "fake")

    dataset = ClassifiedDataset("test-dataset", ["test", "other"])
    dataset.add_dataset([d1, d2, d3, d21, d22, d23])

    for i in range(dataset.get_dataset_size()):
        data = dataset.get_data(i)
        print("{}: class = {}, exp = {}".format(data.get_name(), data.get_class_name(), dataset.get_expected_vector(data.get_class_name())))
