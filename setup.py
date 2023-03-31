from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A Python package to perform unsupervised sentiment analysis using lexicon enhanced Document embeddings'


def readme():
    with open('README.md') as f:
        return f.read()


# Setting up
setup(
    name="lex2sent",
    version=VERSION,
    long_description=readme(),
    author="Kai-Robin Lange",
    author_email="<kai-robin.lange@tu-dortmund.de>",
    description=DESCRIPTION,
    packages=find_packages(),
    url="https://github.com/K-RLange/Lex2Sent",
    install_requires=['nltk', 'gensim', "pandas", "vaderSentiment",
                      "scipy", "numpy", "re", "tqdm"],
    keywords=['python', 'sentiment analysis', 'nlp', 'doc2vec', 'bagging', 'text classification'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
