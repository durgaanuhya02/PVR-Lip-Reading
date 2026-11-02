"""Error-rate metrics: WER, CER, phoneme (PER) and viseme (VER) error rate."""


def edit_distance(ref, hyp):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def align(ref, hyp):
    """Minimum-edit alignment as (ref_item, hyp_item) pairs; None marks a deletion or insertion."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            pairs.append((ref[i - 1], None)); i -= 1
        else:
            pairs.append((None, hyp[j - 1])); j -= 1
    return pairs[::-1]


def error_rate(refs, hyps):
    """Corpus-level error rate = total edits / total reference length."""
    edits = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return edits / max(total, 1)


def wer(ref_texts, hyp_texts):
    return error_rate([r.split() for r in ref_texts], [h.split() for h in hyp_texts])


def cer(ref_texts, hyp_texts):
    return error_rate(list(ref_texts), list(hyp_texts))
