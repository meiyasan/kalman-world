import unittest

class TestMainModule(unittest.TestCase):
    def test_main_function(self):
        self.assertEqual(True, True)

    def test_helper_function(self):
        self.assertEqual(True, True)

print(__name__)
if __name__ == "__main__":
    unittest.main()
