import unittest
from sources.database.connectors.loaders import loader


class TestLoaders(unittest.TestCase):
    def test_load_dataset_tensor(self):
        # Test case for typ = "tensor"
        load_dataset_tensor = loader("tensor")
        self.assertEqual(load_dataset_tensor([1, 2, 3]), [1, 2, 3])

    def test_load_dataset_other(self):
        # Test case for typ = "other"
        with self.assertRaises(Exception):
            load_dataset_other = loader("other")


if __name__ == "__main__":
    unittest.main()
