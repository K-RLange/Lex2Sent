from textClass import *
import unittest

class TestClusterLexicon(unittest.TestCase):

    def test_init(self):
        lexicon = ClusterLexicon()
        self.assertIsInstance(lexicon.full_dict, dict)
        self.assertIsInstance(lexicon.cluster1, list)
        self.assertIsInstance(lexicon.cluster2, list)
        self.assertIsInstance(lexicon.amplifiers, dict)

    def test_check_text(self):
        lexicon = ClusterLexicon()
        self.assertGreater(lexicon.check_text(["very", "good", "day"]), 0)

class TestRatedTexts(unittest.TestCase):
    def test_processing(self):
        texts = RatedTexts(["These are great texts!!!", "Why are you reading this? It's just a test"], ClusterLexicon(),
                           ratings=[1, 0])
        self.assertIsInstance(texts.texts, list)
        self.assertIsInstance(texts.lexicon, ClusterLexicon)
        self.assertIsInstance(texts.texts[0], list)
        self.assertIsInstance(texts.texts[1], list)
        self.assertIsInstance(texts.ratings, list)
        self.assertIsInstance(texts.ratings[0], int)
        self.assertEqual(len(texts.texts), len(texts.ratings))

if __name__ == '__main__':
    unittest.main()