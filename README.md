# Lex2Sent - A bagging approach to unsupervised Sentiment Analysis
Lex2Sent is a text classification/clustering model that can be used with minimal a-priori-information to classify texts into two classes. While the [original paper](https://doi.org/10.48550/arXiv.2209.13023) used it for sentiment analysis on english documents, it is not limited to that purpose, but can be used for any arbitrary type of classification and language as long as there are lexica that can be used as an information-basis.

## Getting Started
You may install this package using either pypi
```
pip install lex2sent
```
or GitHub
```
pip install git+https://github.com/K-RLange/Lex2Sent.git
```

The following is an example of using the Opinion Lexicon to classify an iMDb movie review data set. You may have to use ```nltk.download()``` to download the opinion_lexicon first.
First we configure our data set 
```
from datasets import load_dataset
from nltk.corpus import opinion_lexicon
data = load_dataset('imdb')
ratings, reviews = [], []
for stars, text in zip(data["train"]["label"], data["train"]["text"]):
    if text:
        if stars == 0:
            ratings.append("negative")
        else:
            ratings.append("positive")
        reviews.append(text)
```
And now we can start applying Lex2Sent
```
from lex2sent.textClass import *
lexicon = ClusterLexicon([opinion_lexicon.positive(), opinion_lexicon.negative()])
rated_texts = RatedTexts(reviews, lexicon, ratings)

#Basic "counting" method of classification:
count_res = rated_texts.lexicon_classification_eval(label_list=["positive", "negative"])
l2s_res = rated_texts.lbte(label_list=["positive", "negative"], workers=4)
print("Counting accuracy: {}%; Lex2Sent accuracy: {}%".format(count_res * 100, l2s_res*100))
```
yielding the result "Counting accuracy: 73.772%; Lex2Sent accuracy: 78.172%".

## Reference
Please refer to ["Lex2Sent - A bagging approach to unsupervised Sentiment Analysis"](https://doi.org/10.48550/arXiv.2209.13023) when using this package. When you use this package in a publication, please cite it as
```
@misc{lex2sent,
  title = {{Lex2Sent}: {A} bagging approach to unsupervised sentiment analysis},
	shorttitle = {{Lex2Sent}},
	publisher = {arXiv},
	author = {Lange, Kai-Robin and Rieger, Jonas and Jentsch, Carsten},
	month = sep,
	year = {2022},
	note = {arXiv:2209.13023 [cs]},
	keywords = {Computer Science - Computation and Language},
}
```
## Future Features
-Calling from the console

-FastText and SentenceBERT as alternatives to Doc2Vec

-Options to classify into more than two clusters
