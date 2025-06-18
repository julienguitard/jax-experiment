import unittest
from sources.database.connectors.connexions import Cursor


class TestConnexions(unittest.TestCase):
    def setUp(self):
        self.obj = Cursor()

    def test_query(self):
        # Testing different types of input parameters
        self.assertEqual(self.obj.query(1), 1)
        self.assertEqual(self.obj.query(2.5), 2.5)
        self.assertEqual(self.obj.query("hello"), "hello")
        self.assertEqual(self.obj.query([1, 2, 3]), [1, 2, 3])
        self.assertEqual(self.obj.query({"key": "value"}), {"key": "value"})


if __name__ == "__main__":
    unittest.main()
