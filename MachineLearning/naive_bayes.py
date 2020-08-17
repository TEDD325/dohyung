from collections import Counter, defaultdict
import math, random, re, glob


class NaiveBayesClassifier:

    def __init__(self, training_data, k=0.5):
        self.word_probs = []
        spams = len([label  # number of spam examples
                         for __, label in training_data
                         if label])
        non_spams = len(training_data) - spams  # number of non-spam examples
        word_counts = count_words(training_data)  # word counts
        self.word_probs = [(word,  # for each word
             (spam + k) / (spams + 2 * k),  # P(word | spam)
             (non_spam + k) / (non_spams + 2 * k))  # P(word | ~spam)
             for word, (spam, non_spam) in word_counts.items()]

    def classify(self, example):
        words = tokenize(example)
        log_prob_in_spam = log_prob_in_non_spam = 0.0
        for word, prob_in_spam, prob_in_non_spam in self.word_probs:
            if word in words:  # for each word in the example
                # add the log probability of seeing the word
                log_prob_in_spam += math.log(prob_in_spam)
                log_prob_in_non_spam += math.log(prob_in_non_spam)
            else:  # for each word not in the example
                # add the log probability of not seeing the word
                log_prob_in_spam += math.log(1.0 - prob_in_spam)
                log_prob_in_non_spam += math.log(1.0 - prob_in_non_spam)
        prob_in_spam = math.exp(log_prob_in_spam)
        prob_in_non_spam = math.exp(log_prob_in_non_spam)
        return prob_in_spam / (prob_in_spam + prob_in_non_spam)


def tokenize(example):
    example = example.lower()  # convert to lowercase
    all_words = re.findall("[a-z0-9']+", example)  # extract the words
    return set(all_words)  # remove duplicates


def count_words(training_data):
    counts = defaultdict(lambda: [0, 0])
    for example, label in training_data:
        for word in tokenize(example):
            counts[word][0 if label else 1] += 1
    return counts


def get_data(path):
    data = []
    for fn in glob.glob(path):
        with open(fn, 'r', encoding='ISO-8859-1') as file:
            for line in file:
                index = line.find(" ")
                data.append((line[index + 1:].strip(), eval(line[:index])))
    return data


def split_data(data, prob):
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results


if __name__ == "__main__":
    data = get_data("spam.txt")
    random.seed(0)
    training_data, test_data = split_data(data, 0.75)
    
    classifier = NaiveBayesClassifier(training_data)
    
    classified = [(subject, label, classifier.classify(subject))
              for subject, label in test_data]
    counts = Counter((is_spam, spam_probability > 0.5)  # (actual, predicted)
                     for _, is_spam, spam_probability in classified)
    print("(actual spam, predicted spam) count")
    for e in counts.items():
        print(e[0], e[1])
        
    print()

    classified.sort(key=lambda row: row[2])
    spammiest_nonspams = list(filter(lambda row: not row[1], classified))[-5:]
    nonspammiest_spams = list(filter(lambda row: row[1], classified))[:5]
    print("spammiest non-spams", spammiest_nonspams)
    print("non-spammiest spams", nonspammiest_spams)

    def p_spam_given_word(word_prob):
        __, prob_in_spam, prob_in_non_spam = word_prob
        return prob_in_spam / (prob_in_spam + prob_in_non_spam)

    words = sorted(classifier.word_probs, key=p_spam_given_word)
    print("spammiest words", words[-5:])
    print("non-spammiest words", words[:5])