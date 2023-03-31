from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A Python package to perform unsupervised sentiment analysis using lexicon enhanced Document embeddings'


from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
print(long_description)

# Setting up
setup(
    name="lex2sent",
    version=VERSION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Kai-Robin Lange",
    author_email="<kai-robin.lange@tu-dortmund.de>",
    description=DESCRIPTION,
    packages=find_packages(),
    url="https://github.com/K-RLange/Lex2Sent",
    install_requires=['nltk', 'gensim', "pandas", "vaderSentiment",
                      "scipy", "numpy", "tqdm"],
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
