import re
from collections import defaultdict


def map_wc(document):
    for word in tokenize(document):  # each word in document
        yield (word, 1)  # output (word, 1)


def reduce_wc(word, counts):
    yield (word, sum(counts))  # output (word, sum of counts)


def map_reduce(input_objects, mapper, reducer):
    collector = defaultdict(list)
    for data_object in input_objects:
        for key, value in mapper(data_object):
            collector[key].append(value)
    return [output_object
            for key, counts in collector.items()
            for output_object in reducer(key, counts)]


def tokenize(document):
    document = document.lower()
    all_words = re.findall("[a-z0-9']+", document)
    return set(all_words)


if __name__ == "__main__":
    documents = ["the white rabbit", "the black turtle", "the rabbit and the turtle"]
    print(map_reduce(documents, map_wc, reduce_wc))
    print()
    
