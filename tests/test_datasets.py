import unittest

from voicemap.datasets import ClassConcatDataset, DummyDataset


class TestClassConcatDataset(unittest.TestCase):
    def test_dataset(self):
        data1 = DummyDataset(1, 10)
        data2 = DummyDataset(1, 5)

        data = ClassConcatDataset([data1, data2])

        self.assertEqual(data.num_classes, data1.num_classes + data2.num_classes)

        class_indicies = []
        for i, (x, y) in enumerate(data):
            print(i, (x.shape, y))
            class_indicies.append(y)

        self.assertEqual(min(class_indicies), 0)
        self.assertEqual(max(class_indicies), data.num_classes - 1)
