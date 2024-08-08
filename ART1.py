import numpy as np

class ART1:
    def __init__(self, num_features, vigilance):
        self.num_features = num_features
        self.vigilance = vigilance
        self.categories = []

    def match_category(self, input_pattern):
        for i, category in enumerate(self.categories):
            match_score = np.sum(np.minimum(input_pattern, category)) / np.sum(input_pattern)
            if match_score >= self.vigilance:
                return i
        return -1

    def train(self, input_patterns):
        for pattern in input_patterns:
            category_idx = self.match_category(pattern)
            if category_idx == -1:
                self.categories.append(pattern)
            else:
                self.categories[category_idx] = np.minimum(self.categories[category_idx], pattern)

    def classify(self, input_pattern):
        category_idx = self.match_category(input_pattern)
        return category_idx if category_idx != -1 else "No match"

# Example usage
input_patterns = np.array([
    [1, 1, 0, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 0, 1, 1]
])

art1 = ART1(num_features=4, vigilance=0.75)
art1.train(input_patterns)

# Test the network with a new pattern
test_pattern = np.array([1, 1, 0, 1])
#test_pattern = np.array([1, 1, 0, 0])
category = art1.classify(test_pattern)
print(f"Test Pattern classified into category: {category}")
