from random import random, shuffle


def new_choices(population, k=1):
    """
    Draws samples with replacement.
    Args:
        population: list. Population of which a Bootstrap is to be drawn.
        k: int. Number of samples to draw.
    Returns:
        k-dimensional list containing random samples from population.
    """
    _int = int
    if type(population) is not list:
        population = [population]
    total = len(population)
    return [population[_int(random() * total)] for i in range(k)]


def bnf_dict_creation(text, window=3, forward=False):
    """
    Creates a dict of context words for every word in a text given a window size
    Args:
        text: list of strings. The text (with correct sentence structure) that
              is to be resampled.
        window: int. Window size
        forward: boolean. Is the window only expanding in front or also behind
                 the current word?
    Returns:
        dict. Contains the context words of each word.
    """
    set_of_words = dict(zip(text, [[] for x in range(len(text))]))
    for inner_index in range(len(text)):
        word = text[inner_index]
        if forward:
            distance = 1
            while distance <= window:
                if inner_index + distance < len(text):
                    set_of_words[word].append(text[inner_index + distance])
                else:
                    set_of_words[word].append(
                        text[inner_index + distance - len(text)]
                    )
                distance = distance + 1
        else:
            distance = -window
            while (distance <= window) & (inner_index + distance < len(text)):
                if (distance != 0) & (inner_index + distance >= 0):
                    set_of_words[word].append(text[inner_index + distance])
                distance = distance + 1
    return set_of_words


def bnf_resampling(texts, forward=True):
    """
    Executes a markov-text-resampling based on the context of each word. After
    sampling one word, one of the three words that follow up on it in the
    original text is sampled.
    Args:
        texts: list of lists of strings. Each list represents a text and
               each string represents a word.
        forward: boolean. Should the window exceed only after the current
                 word (True, default) or should it exceed in both directions
                 of the text (False)?
    Returns:
        list of lists of strings. Resampled texts.
    """
    new_texts = [[] for text in texts]
    for index in range(len(texts)):
        text = texts[index]
        latest = None
        word_list = bnf_dict_creation(text, forward=forward)
        for word in text:
            if latest is None:
                latest = new_choices(list(word_list.keys()))[0]
            else:
                latest = new_choices(word_list[latest])[0]
            new_texts[index].append(latest)
    return new_texts


def bw_resampling(texts):
    """
    Applies simple inner-text word-based bootstrap to a list of texts.
    Args:
        texts: list of lists of strings. Each list represents a text and
                each string represents a word.
    Returns:
        list of lists of strings. Resampled texts.
    """
    new_texts = [[] for text in texts]
    for current_text in range(len(texts)):
        text = texts[current_text]
        new_texts[current_text] = new_choices(text, len(text))
    return new_texts


def bwp_resampling(texts):
    """
    Permutes all words inside each text.
    Args:
        texts: list of lists of strings. Each list represents a text and
                each string represents a word.
    Returns:
        list of lists of strings. Resampled texts.
    """
    new_texts = []
    for current_text in range(len(texts)):
        text = texts[current_text]
        shuffle(text)
        new_texts.append(text)
    return new_texts
