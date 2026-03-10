def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    s1 = set(set_a)
    s2 = set(set_b)
    intersection = s1 & s2
    union = s1 | s2
    if len(union) == 0:
        return 0.0
    j = len(intersection) / len(union)
    return float(j)
    