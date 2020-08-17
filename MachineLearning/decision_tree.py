from collections import Counter, defaultdict
from functools import partial
import math


def construct_tree(training_data, attributes=None):
    if attributes is None:  # if the first pass
        attributes = training_data[0][0].keys()  # use all attributes
    num_trues = len([label for __, label in training_data if label])  # number of Trues
    num_falses = len(training_data) - num_trues  # number of Falses
    if num_trues == 0:  # if only Falses are left
        return False  # return a "False" leaf
    if num_falses == 0:  # if only Trues are left
        return True  # return a "True" leaf
    if not attributes:  # if no attribute left
        return num_trues >= num_falses  # return the majority leaf
    best_attribute = min(attributes,  #  the best attribute for splitting
        key=partial(conditional_entropy, training_data))
    groups = group_by(training_data, best_attribute)  # group by the best attribute
    remaining_attributes = [a for a in attributes if a != best_attribute]
    subtrees = { attribute : construct_tree(group, remaining_attributes)  # construct subtrees
                 for attribute, group in groups.items() }
    subtrees[None] = num_trues > num_falses  # default case
    return (best_attribute, subtrees)  # return the internal node


def classify(tree, example):
    if tree in [True, False]:  # if a leaf node
        return tree
    attribute, subtree = tree  # internal node
    key = example.get(attribute)  # attribute value from the example
    if key not in subtree:  # if no subtree for the attribute value
        key = None  # the None subtree is used
    subtree = subtree[key]  # the subtree to visit
    return classify(subtree, example)  # continue recursively


def conditional_entropy(training_data, attribute):
    groups = group_by(training_data, attribute).values()
    total_count = sum(len(group) for group in groups)
    return sum(entropy(group) * len(group) / total_count
        for group in groups)


def group_by(training_data, attribute):
    groups = defaultdict(list)
    for example in training_data:  # each example
        v = example[0][attribute]  # attribute value from the example
        groups[v].append(example)  # register the example
    return groups


def entropy(training_examples):
    labels = [label for _, label in training_examples]
    probabilities = label_probabilities(labels);
    return sum(-p * math.log(p, 2) for p in probabilities if p)


def label_probabilities(labels):
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]


if __name__ == "__main__":

    training_data = [
        ({'age':'young', 'income':'high', 'student':'no', 'credit_rating':'fair'}, False),
        ({'age':'young', 'income':'high', 'student':'no', 'credit_rating':'excellent'}, False),
        ({'age':'middle', 'income':'high', 'student':'no', 'credit_rating':'fair'}, True),
        ({'age':'senior', 'income':'medium', 'student':'no', 'credit_rating':'fair'}, True),
        ({'age':'senior', 'income':'low', 'student':'yes', 'credit_rating':'fair'}, True),
        ({'age':'senior', 'income':'low', 'student':'yes', 'credit_rating':'excellent'}, False),
        ({'age':'middle', 'income':'low', 'student':'yes', 'credit_rating':'excellent'}, True),
        ({'age':'young', 'income':'medium', 'student':'no', 'credit_rating':'fair'}, False),
        ({'age':'young', 'income':'low', 'student':'yes', 'credit_rating':'fair'}, True),
        ({'age':'senior', 'income':'medium', 'student':'yes', 'credit_rating':'fair'}, True),
        ({'age':'young', 'income':'medium', 'student':'yes', 'credit_rating':'excellent'}, True),
        ({'age':'middle', 'income':'medium', 'student':'no', 'credit_rating':'excellent'}, True),
        ({'age':'middle', 'income':'high', 'student':'yes', 'credit_rating':'fair'}, True),
        ({'age':'senior', 'income':'medium', 'student':'no', 'credit_rating':'excellent'}, False)
    ]

    print("conditional entropy") # 어떤 attribute을 알 때 Entropy가 낮아지는지를 계산하여 보여준다. 불확실성이 제일 낮게 나오는 attribute은 age이므로 decision tree에서 age부터 물어봐야 한다. 나이가 불확실성을 가장 낮춘다.
    for attribute in ['age', 'income', 'student', 'credit_rating']:
        print(attribute, conditional_entropy(training_data, attribute))
    print()

    tree = construct_tree(training_data)
    print("decision tree")
    print(tree)
    print()
    
    print("test")
    for example in training_data:
        print(example[0], classify(tree, example[0]))
    print()

    print("senior", classify(tree, { "age" : "senior" })) 
    print("middle", classify(tree, { "age" : "middle" }))
    print("young", classify(tree, { "age" : "young" }))
    print("unknown", classify(tree, { "age" : "unknown" }))