import pickle
import re
import logging
from tqdm import tqdm
import warnings
from vaderSentiment import vaderSentiment
import io
import operator
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import Bootstrap
from itertools import product
import pandas as pd
from random import seed, sample
from scipy import spatial
import numpy as np


class ClusterLexicon:
    """
    Provides functionalities for lexica. Negations, and amplifiers
    can be included. The lexicon clusters cluster1 and cluster2
    are included to provide functionalities for lexicon-based text embedding
    methods which are defined in the RatedTexts class.
    """

    full_dict = {}
    cluster1 = []
    cluster2 = []
    amplifiers = {}

    def __init__(self, lexicon=None, negations=None, amplifiers=None):
        """
        Initializes the Lexicon using a dict or list of lists of
        cluster words and possibly lists of negations and amplifiers
        Args:
            lexicon: dict or list of lists. The lexicon that is to be used.
            negations: list of strings. Words that are to be considered as
                       negations.
            amplifiers: list of strings. Words that are to be considered as
                        amplifiers.
        Returns:
            None
        """
        if amplifiers is None:
            amplifiers = vaderSentiment.BOOSTER_DICT
        if negations is None:
            negations = vaderSentiment.NEGATE
        if lexicon is None:
            lexicon = vaderSentiment.SentimentIntensityAnalyzer().lexicon
        if isinstance(lexicon, dict):
            self.full_dict = lexicon
            self.full_dict = {
                x: self.full_dict[x]
                for x in self.full_dict.keys()
                if self.full_dict[x] != 0
            }
        elif isinstance(lexicon, list):
            self.full_dict = dict(
                zip(lexicon[0], [1 for x in range(len(lexicon[0]))])
            )
            self.full_dict.update(
                dict(zip(lexicon[1], [-1 for x in range(len(lexicon[1]))]))
            )
        if isinstance(amplifiers, dict):
            self.amplifiers = amplifiers
        if negations:
            self.full_dict.update(
                {
                    "neg" + str(key): -0.5 * value
                    for key, value in self.full_dict.items()
                }
            )

        self.cluster1 = [
            key for key, value in self.full_dict.items()
            if value > 0 and isinstance(key, str)
        ]
        self.cluster2 = [
            key for key, value in self.full_dict.items()
            if value < 0 and isinstance(key, str)
        ]

        self.amplifiers = dict(
            zip(amplifiers, [2 for x in range(len(amplifiers))])
        )
        self.amplifiers.update(
            dict(zip(negations, [-0.5 for x in range(len(negations))]))
        )

    def __getitem__(self, item):
        """
        Returns the value of a word if it is part of the lexicon.
        Args:
            item: string. The word that is to be searched.
        Returns:
            The lexicon value, amplifier value for the word or an
            exception if the word is not part of the lexicon.
        """
        if item in self.full_dict:
            return self.full_dict[item]
        elif item in self.amplifiers:
            print("amplifier value: " + str(self.amplifiers[item]))
        else:
            raise Exception("The chosen item is not part of the lexicon")

    def __contains__(self, item):
        """
        Checks if a word is part of the lexicon
        Args:
            item: string. The word that is to be searched.
        Returns:
            True if the word is part of the lexicon, False otherwise.
        """
        if item in self.full_dict or item in self.amplifiers:
            return True
        else:
            return False

    def get_negations(self):
        """
        Returns the negations of the lexicon.
        """
        return [key for key, value in self.amplifiers.items() if value < 1]

    def get_amplifiers(self):
        """
        Returns the amplifiers of the lexicon
        """
        return [key for key, value in self.amplifiers.items() if value > 1]

    def add_amplifiers(self, amplifiers):
        """
        Adds a list of amplifiers to the lexicon.
        Args:
            amplifiers: list of strings. The amplifiers that are to be added.
        Returns:
            None
        """
        self.amplifiers.update(
            dict(zip(amplifiers, [2 for x in range(len(amplifiers))]))
        )

    def add_negations(self, negations):
        """
        Adds a list of negations to the lexicon without adding "neg"+word to
        the lexicon halves.
        Args:
            negations: list of strings. The negations that are to be added.
        Returns:
            None
        """
        self.amplifiers.update(
            dict(zip(negations, [-0.5 for x in range(len(negations))]))
        )

    def check_text(self, text):
        """
        Creates a classifier value of the given text by "counting" cluster
        words within. Negations and amplifiers are also used if
        provided within the class variables.
        Args:
            text: list of strings. The text that is to be analyzed.
        Returns:
            The classifier value of the text. Negative values indicate a
            text belonging to cluster 1.
        """
        if not self.amplifiers:
            return sum(
                list(
                    map(
                        lambda word: 0
                        if word not in self.full_dict
                        else self.full_dict[word],
                        text,
                    )
                )
            )
        else:
            amplifier = 1
            sent_classifier = 0
            for x in range(len(text)):
                word = text[x]
                if word in self.full_dict:
                    sent_classifier += self.full_dict[word] * amplifier
                    amplifier = 1
                elif word in self.amplifiers:
                    amplifier = self.amplifiers[word]
                else:
                    amplifier = 1
            return sent_classifier

    def count(self, text):
        """
        Returns a "counting" value of the lexicon, displaying the difference
        in occurrences of cluster words in a text.
        Args:
            text: list of strings. The text that is to be analyzed.
        Returns:
            The "counting" value of the text. Negative values indicate a
            text belonging to cluster 2.
        """
        return sum(
            list(
                map(
                    lambda word: 0 if word not in self.full_dict
                    else 2 * int(self.full_dict[word] > 0) - 1,
                    text,
                )
            )
        )

    def copy(self):
        """
        Returns a copy of the lexicon.
        """
        return ClusterLexicon(self.full_dict.copy(),
                              amplifiers=self.amplifiers.copy())


class RatedTexts:
    """
    Eases the process of unsupervised text clustering using lexica.
    Unprocessed texts, as well as a lexicon object of the class
    ClusterLexicon can be used as inputs to create processed texts that use
    lemmatization as well as stopword deletion. Lexicon-based text embedding
    methods and traditional lexicon methods can be performed using the
    functions lbte() and lexicon_classification() respecitvely.
    """

    texts = []
    unprocessed_texts = []
    lexicon = None
    ratings = []
    kwargs = {}
    number_of_texts = len(texts)

    def __init__(
            self,
            unprocessed_texts=None,
            lexicon=None,
            ratings=None,
            texts=None,
            label_list=None,
            **kwargs
    ):
        """
        Initializes the RatedTexts object
        Args:
            unprocessed_texts: list of strings. Each string is one document.
            lexicon: Lexicon object of the ClusterLexicon class. Is used for
                     "counting" and as a basis for lexicon-based text embeddings.
            ratings: list of strings. Labels for the given texts. Is needed if a
                     classification rate is to be determined.
            texts: list of list of strings. Each element of the major list
                   represents a text, which is a list of strings representing
                   the words. Use this parameter if you want to use your own
                   preprocessing.
            label_list: list of strings. The labels that are to be used for
                        classification. If None, the labels will be determined
                        automatically.
            kwargs: Additional parameters used for the preprocessing method
                    "process_texts".
        Returns:
            None
        """
        if lexicon is None:
            warnings.warn(
                "No lexicon provided. The default lexicon 'VADER' will be used."
            )
            lexicon = ClusterLexicon()
        if label_list is None:
            self.label_list = list(set(ratings))
        if unprocessed_texts is None and texts is None:
            warnings.warn("No texts provided. Initializing texts as None.")
        self.unprocessed_texts = unprocessed_texts
        self.lexicon = lexicon
        self.kwargs = kwargs
        if texts is None and unprocessed_texts is not None:
            self.texts = self.process_texts(unprocessed_texts, **kwargs)
        else:
            self.texts = texts
        self.ratings = ratings
        self.number_of_texts = len(self.texts)

    def __getitem__(self, index):
        """
        Returns the text with a given index
        Args:
            index: int. Index of the text
        Returns:
            The preprocessed text with the given index as one string or the
            rating and the preprocessed text if available.
        """
        try:
            return " ".join([self.ratings[index]] + self.texts[index])
        except IndexError as e:
            return " ".join(self.texts[index])

    def process_texts(self, texts, **kwargs):
        """
        Applies preprocessing to the unprocessed texts. Lemmatization,
        tokenization, punctuation removal and stopword removal is applied.
        Words included in the lexicon are not used as stop words.
        Args:
            texts: list of strings. Texts that are to be processed.
            kwargs: additional arguments. Can include "lemmatizer",
                    "tokenizer" and "stop_words" to change the default options
        Returns:
            Tokenized and preprocessed texts as a list of lists of strings
        """
        logging.info("Starting preprocessing...")
        if "lemmatizer" in kwargs:
            lemmatizer = kwargs["lemmatizer"]
        else:
            lemmatizer = WordNetLemmatizer()
        if "tokenizer" in kwargs:
            tokenizer = kwargs["tokenizer"]
        else:
            tokenizer = RegexpTokenizer(r"\w+")
        if "stop_words" in kwargs:
            stop_words = kwargs["stop_words"]
        else:
            stop_words = {
                "".join(
                    [
                        character
                        for character in word
                        if character.isalnum() or character.isspace()
                    ]
                )
                for word in stopwords.words("english")
            }
            if self.lexicon is not None:
                stop_words = {
                    word for word in stop_words if word not in self.lexicon
                }

        if not isinstance(texts, list):
            raise Exception("The texts-object must be a list of strings!")
        try:
            if not isinstance(texts[0], list):
                texts = list(map(lambda x: re.sub(r"[^a-z\s]", " ", x.lower()), texts))
                texts = list(map(word_tokenize, texts))
        except TypeError:
            print(
                """The texts-object has to be either a list of strings or a 
                list of lists of strings!"""
            )
            raise
        processed_texts = [[] for x in texts]
        for x in range(len(texts)):
            texts[x] = [word for word in texts[x] if word
                        not in stop_words]
            texts[x] = [
                lemmatizer.lemmatize(word) for word in texts[x]
            ]
            texts[x] = [word for word in texts[x] if word not in stop_words]
            negations = self.lexicon.get_negations()
            if negations:
                gets_negated = False
                for y in range(len(texts[x])):
                    if gets_negated:
                        processed_texts[x].append("neg" + texts[x][y])
                        gets_negated = False
                    elif texts[x][y] in negations:
                        gets_negated = True
                    else:
                        processed_texts[x].append(texts[x][y])
            else:
                processed_texts[x] = texts[x]
        return processed_texts

    def add_lexicon(self, lexicon):
        """
        Adds a lexicon of choice to the RatedTexts object (does not change the
        processing of texts done before)
        Args:
            lexicon: Lexicon object of the ClusterLexicon class.
        Returns:
            None
        """
        self.lexicon = lexicon

    def get_classification_rate(self, prediction, index=None):
        """
        Returns the classification rate of given predicted labels.
        Args:
            prediction: list. Predicted labels
            index: list. Index of the labels that should be compared.
                   Optional and only used if not the entire dataset
                   should be compared.
        Returns:
            The classification rate as a float.
        """
        if index is None:
            if len(prediction) != self.number_of_texts:
                raise Exception(
                    "Length of prediction does not match number of texts!"
                )
            return sum(
                [
                    self.ratings[x] == prediction[x]
                    for x in range(self.number_of_texts)
                ]
            ) / len(prediction)
        else:
            sub_ratings = operator.itemgetter(*index)(self.ratings)
            return sum(
                [
                    sub_ratings[x] == prediction[x]
                    for x in range(len(sub_ratings))
                ]
            ) / len(sub_ratings)

    def texts_as_string(self):
        """
        Returns the processed texts as a single string each.
        """
        return list(map(" ".join, self.texts))

    def write_to_file(self, file, prediction=None, index=None):
        """
        Writes the corpus and given labels to a certain path as a .txt-file
        Args:
            file: str. Path to write the files to
            prediction: list. Predicted labels. If None, the true labels
                        (self.ratings) are used.
            index: int. Is used if only a part of the corpus is supposed to
                   be saved.
        Returns:
            None
        """
        if prediction is None:
            prediction = self.ratings
        full_texts = self.texts_as_string()
        if index is None:
            new_texts = [
                prediction[x] + " " + full_texts[x] + " \n"
                for x in range(self.number_of_texts)
            ]
        else:
            sub_texts = operator.itemgetter(*index)(self.texts)
            new_texts = [
                prediction[x] + " " + " ".join(sub_texts[x]) + " \n"
                for x in range(len(prediction))
            ]
        with io.open(file, "w", encoding="utf-8") as f:
            for x in new_texts:
                f.write(x)

    def to_pickle(self, file):
        """
        Saves the RatedTexts object as a pickle-file
        Args:
            file: Path to save the object in
        Returns:
            None
        """
        with open(file, "wb") as output_file:
            pickle.dump(self, output_file)

    def update(self, texts=None, ratings=None, unprocessed_texts=None):
        """
        Replaces the class variables with other objects.
        """
        if texts is not None:
            self.texts = texts
            self.number_of_texts = len(texts)
        if ratings is not None:
            self.ratings = ratings
        if unprocessed_texts is not None:
            self.unprocessed_texts = unprocessed_texts
        if self.number_of_texts != len(self.unprocessed_texts):
            raise Exception(
                "The number of texts and unprocessed texts must be identical."
            )

    def copy(self):
        """
        Copies the RatedTexts objects and returns it.
        """
        return RatedTexts(
            self.unprocessed_texts.copy(),
            self.lexicon.copy(),
            self.ratings.copy(),
            self.texts.copy()
        )

    def reduce_size(self, new_size):
        """
        Reduces the corpus size to a certain percentage. A deterministic
        version of draw_subsamples.
        Args:
            new_size: Percantage of the size of the old corpus.
        Returns:
            None
        """
        self.number_of_texts = round(self.number_of_texts * new_size)
        self.texts = self.texts[0:self.number_of_texts]
        self.ratings = self.ratings[0:self.number_of_texts]

    def draw_subsamples(self, percent):
        """
        Randomly draws subsamples of the texts from the corpus to a certain
        percentage. Each text is chosen randomly.
        Args:
            percent: Percantage of the size of the old corpus.
        Returns:
            None
        """
        indices = sample(
            range(self.number_of_texts),
            round(self.number_of_texts * percent / 100),
        )
        self.texts = list(map(lambda x: self.texts[x], indices))
        self.ratings = list(map(lambda x: self.ratings[x], indices))
        if len(self.unprocessed_texts) > 0:
            self.unprocessed_texts = list(
                map(lambda x: self.unprocessed_texts[x], indices)
            )
        self.number_of_texts = round(self.number_of_texts * percent / 100)

    def lexicon_classification(self, lexicon=None):
        """
        Creates a classifier by using a lexicon "counting" method. Either uses
        the lexicon saved within this RatedTexts
        object or one provided as an input.
        Args:
            lexicon: Object of the ClusterLexicon class. Is used as to
                     classify the texts by.
        Returns:
            List of floats representing lexicon "counting" values to label
            or sort the texts by.
        """
        if self.lexicon is None and lexicon is None:
            raise Exception(
                """The RatedTexts object must either contain a ClusterLexicon
                object or it must be an input to this function!"""
            )
        elif isinstance(lexicon, ClusterLexicon):
            return list(map(lambda x: lexicon.check_text(x), self.texts))
        elif lexicon is not None:
            return list(map(lambda x: lexicon(x), self.unprocessed_texts))
        else:
            return list(map(lambda x: self.lexicon.check_text(x), self.texts))

    def lexicon_classification_eval(self, lexicon=None, label_list=None):
        """
        Evaluates the lexicon classification by using the true labels.
        Returns:
            The classification rate as a float between [0,1].
        """
        if self.ratings is None:
            raise Exception(
                """The RatedTexts object must contain the true labels to
                evaluate the lexicon classification!"""
            )
        else:
            counts = self.lexicon_classification(lexicon)
            if label_list is None:
                label_list = self.label_list
            sentiment, chosen_texts = self.sent_for_threshold(counts, label_list=label_list)
            return self.get_classification_rate(sentiment, chosen_texts)

    def lbte(
            self,
            resampling=Bootstrap.bw_resampling,
            rng_seed=0,
            threshold=0,
            grid=None,
            sorting="absolute",
            pre_resampling_sorting=False,
            label_list=None,
            path="",
            verbose=True,
            workers=1,
    ):
        """
        Performs unsupervised sentiment analysis using lexicon-based text
        embeddings. Returns the classification rate as a float between [0,1].
        Currently only Doc2Vec is implemented.
        Args:
            resampling: Resampling procedure used to resample the corpus
            rng_seed: Seed for the random number generator
            threshold: float or list of length 2. threshold or cutoff to
                          seperate the labels by. Can be a float in [0,1),
                          representing either a quantile of the classifier
                          resulting from the lbte or the fixed value 0. Can
                          also be a list of floats in (0,1) to classify only
                          documents whose classifier value exceeds the
                          quantiles.
            grid: dict. Grid to perform the lbte on, provided as a dict. The
                  keys represent the parameters inside the text embedding
                  model and the values, must be provided as lists, determine
                  the parameter values.
            sorting: str. Way of sorting the documents based on the value that
                     results from lexicon-checking. Can be either "ascending",
                     "descending" or "absolute" (sorting highest absolute
                     values to smallest).
            pre_resampling_sorting: boolean. Should the values be sorted before
                                    resampling or after?
            label_list: list of ints. List of labels to use for the lbte. If
                        None, the labels are generated automatically.
            path: str. Path to save the model in.
            verbose: boolean. Should the progress be printed?
        Returns:
            float. Classification rate.
        """
        if label_list is None:
            label_list = self.label_list
        if grid is None:
            grid = {
                "epochs": [5, 10, 15],
                "window": [5, 10, 15],
                "vector_size": [64, 128, 256],
            }
        if not self.lexicon:
            raise Exception(
                "A lexicon-base is needed for lexicon-based text embeddings!"
            )
        if not self.texts:
            raise Exception(
                "The texts that are to be analyzed need to be defined!"
            )

        if pre_resampling_sorting:
            lex_count = list(
                map(lambda text: self.lexicon.count(text), self.texts)
            )

        if isinstance(grid, dict):
            diff_dist_all = []
            grid_df = pd.DataFrame(
                [row for row in product(*grid.values())], columns=grid.keys()
            )
            logging.info(
                str(len(grid_df)) + " different parameter-combinations found!"
            )
            seed(rng_seed)
            counter = 1
            if verbose:
                iterator = tqdm(grid_df.iterrows(), total=len(grid_df))
                logging.info("Starting training...")
            else:
                iterator = grid_df.iterrows()
            for combination in iterator:
                new_texts = self.texts.copy()
                new_texts = resampling(new_texts)
                kwargs = combination[1]
                eval_texts = new_texts.copy()
                if not pre_resampling_sorting:
                    lex_count = list(
                        map(
                            lambda text: self.lexicon.count(text),
                            eval_texts,
                        )
                    )

                if sorting == "ascending":
                    eval_texts = [
                        x for _, x in sorted(zip(lex_count, eval_texts))
                    ]
                elif sorting == "decending":
                    eval_texts = [
                        x for _, x in sorted(zip(lex_count, eval_texts))
                    ]
                    eval_texts.reverse()
                elif sorting == "absolute":
                    eval_texts = [
                        x
                        for _, x in sorted(
                            zip(list(map(abs, lex_count)), eval_texts)
                        )
                    ]
                    eval_texts.reverse()
                docs = [
                    TaggedDocument(doc, [i])
                    for i, doc in enumerate(eval_texts)
                ]
                dc_model = Doc2Vec(docs, workers=workers, **kwargs)
                pos_emb = dc_model.infer_vector(self.lexicon.cluster1)
                neg_emb = dc_model.infer_vector(self.lexicon.cluster2)
                embeddings = list(
                    map(lambda x: dc_model.infer_vector(x), new_texts)
                )
                diff_dist_all.append(
                    list(
                        map(
                            lambda x: spatial.distance.cosine(x, neg_emb)
                                      - spatial.distance.cosine(x, pos_emb),
                            embeddings,
                            )
                    )
                )
                counter += 1
            diff_dist = list(
                map(
                    lambda y: sum(map(lambda x: x[y], diff_dist_all)),
                    range(self.number_of_texts),
                )
            )
            sentiment, chosen_texts = self.sent_for_threshold(
                diff_dist, threshold, label_list
            )
            if path != "":
                self.write_to_file(path, sentiment, chosen_texts)
            return self.get_classification_rate(sentiment, chosen_texts)
        else:
            raise Exception(
                """The grid must be a dict object, containing the parameters as
                the keys as well as their values as the dict values."""
            )

    def sent_for_threshold(
            self,
            classifier,
            threshold=0,
            label_list=None,
    ):
        """
        Assigns labels based on the values saved inside a classifier
        variable. Values larger than a certain threshold are assigned the first
        label from the label_list. This threshold can be the fixed value 0, a
        quantile or split in two separate thresholds. If only one threshold is
        chosen, every document gets labeled. If the classifier value of a
        document equals said threshold, the label is chosen randomly.
        Args:
            classifier: Classifier value to determine the labels by.
            threshold: Threshold or cutoff by which the labels are separated
            label_list: list of labels that are to be assigned.
        Returns:
            labels and chosen_texts. The assigned labels are saved inside
            the labels vector. The chosen_texts vector is only relevant if two
            separate thresholds are chosen.
        """
        if label_list is None:
            label_list = self.label_list
        if threshold == 0:
            threshold = [0, 0]
        elif not isinstance(threshold, list):
            threshold = np.quantile(classifier, [threshold, 1 - threshold])
        else:
            threshold = np.quantile(classifier, threshold)
        labels = []
        chosen_texts = []
        for x in range(0, len(classifier)):
            var = classifier[x]
            if var > threshold[1]:
                labels.append(label_list[0])
                chosen_texts.append(x)
            elif var < threshold[0]:
                labels.append(label_list[1])
                chosen_texts.append(x)
            elif var == threshold[1] and var == threshold[0]:
                labels.append(sample(label_list, 1)[0])
                chosen_texts.append(x)
        return labels, chosen_texts