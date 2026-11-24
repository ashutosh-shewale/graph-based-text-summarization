# How it works

The code is split so that each stage can be explained, tested and swapped on its own. `pipeline.Document` runs preprocessing, TF-IDF and PageRank once. All three summarizers then reuse those results.

Notation: N = number of sentences, V = vocabulary size, n = summary length, K = number of clusters.

---

## 1. Preprocessing ([`preprocessing.py`](../summarizer/preprocessing.py))

Each sentence is stored as a `Sentence` with three views:

| Field | Example (input "I was walking down the street.") | Used by |
|---|---|---|
| `original` | `I was walking down the street.` | Final summary output |
| `cleaned_tokens` | `i was walking down the street` | Word graph (keeps function words so fused sentences read naturally) |
| `processed_tokens` | `walk street` | TF-IDF |

The steps follow the spec: remove punctuation with `str.translate`, lowercase, run `word_tokenize`, drop NLTK stopwords, then apply the WordNet lemmatizer.

**POS-aware lemmatization.** `WordNetLemmatizer` assumes every word is a noun unless it is told otherwise. As a result, "walking" stays "walking" and "met" stays "met". The spec's example expects "walk" and "meet", so each token is POS-tagged first (`nltk.pos_tag`) and the tag is mapped to a WordNet POS. Tagging runs before stopword removal so the tagger still sees the full sentence context.

**Punctuation.** `string.punctuation` only covers ASCII. The sample texts contain curly quotes and dashes, so those characters are added to the removal table.

**Sentence splitting.** By default the input is split on newlines, as the spec says. For text that wraps across lines, `--split sentence` uses NLTK's Punkt model instead.

## 2. TF-IDF ([`tfidf.py`](../summarizer/tfidf.py))

This is written from scratch with NumPy, using the spec's exact formula:

```
TF(w, s)  = count of w in s
IDF(w)    = log(N / df(w))
TFIDF     = TF × IDF          → matrix of shape (V words, N sentences)
```

sklearn's `TfidfVectorizer` isn't used here: it adds 1 to IDF (smoothing) and L2-normalises rows, so it would not match the spec. A word that appears in every sentence gets IDF = log(1) = 0. It can't tell sentences apart, so it carries no weight.

Cosine similarity between the sentence vectors (the columns) comes from `sklearn.metrics.pairwise.cosine_similarity`. A sentence whose vector is all zeros gets similarity 0 with everything instead of a division-by-zero error.

**Cost:** O(N·V) to build the matrix and O(N²·V) for all pairwise similarities.

## 3. PageRank ([`pagerank.py`](../summarizer/pagerank.py))

- Build a complete undirected graph with one node per sentence. Each edge weight is the cosine similarity between the two sentences.
- **No self-loops.** Every sentence has similarity 1 with itself. If those self-loops were kept, each node would mostly "vote" for itself, and the scores would flatten towards uniform. The original code had this bug.
- Run `networkx.pagerank(graph, alpha=0.85, weight="weight")`. A sentence scores highly when it is similar to many sentences that are themselves high-scoring, which makes it *central* to the document. This is the idea behind TextRank and LexRank.
- Take the top-n sentences and **re-sort them by position in the document** so the summary reads in order.

**Cost:** about O(iterations · N²), because the graph is complete.

## 4. MMR ([`mmr.py`](../summarizer/mmr.py))

The selection is greedy. Each round picks the remaining sentence with the highest value of

```
λ · Imp(Di) − (1 − λ) · max over already-selected Dj of Sim(Di, Dj)
```

- `Imp` is the PageRank score, **min-max scaled to [0, 1]**. Raw PageRank scores add up to 1, so each is tiny (about 1/N) next to cosine similarities in [0, 1]. Without the rescaling, the redundancy penalty would always win and λ would have no real effect.
- `Sim` is the TF-IDF cosine similarity. The penalty uses the **maximum** over *all* sentences already selected. The original code compared each candidate only with the last pick, and compared characters rather than words.
- λ = 1 gives plain PageRank; λ = 0 picks for diversity only.

**Cost:** O(n² · N).

On the inaugural-speech and galaxy samples, the top PageRank sentences are already dissimilar to each other (cosine ≤ 0.18), so MMR leaves the selection unchanged. That is the correct result. `data/river_flood_news.txt` contains deliberate paraphrases, which shows MMR actually removing one.

## 5. K-means ([`clustering.py`](../summarizer/clustering.py))

The from-scratch version follows the spec:

1. Choose K sentences with `random.sample` as the initial centroids.
2. Assign each sentence to the centroid with the highest cosine similarity.
3. Set each centroid to the mean of its members.
4. Repeat until the assignments no longer change. A `max_iterations` cap guarantees the loop ends.

If a cluster loses all its members, it **keeps its previous centroid**. The mean of an empty set is NaN, and in the original code that NaN spread to every later step. The original code also set K = N, so every sentence ended up in its own cluster. Here K equals the requested summary length, so each cluster contributes one sentence.

`--kmeans sklearn` runs `sklearn.cluster.KMeans` on L2-normalised vectors as a baseline. For unit vectors, ‖a − b‖² = 2 − 2·cos(a, b), so Euclidean k-means groups points the same way cosine-based k-means does.

**Cost:** O(iterations · N · K · V).

## 6. Word-graph sentence fusion ([`sentence_graph.py`](../summarizer/sentence_graph.py))

For each cluster:

1. **S1** is the member closest to the centroid by cosine similarity. Ties are broken at random.
2. **S2** is the member that shares the most bigrams with S1, and it must share at least 3. If no member qualifies, S1 is output unchanged.
3. Build a directed graph `START → b1 → b2 → … → bm → END` for both sentences. Because nodes are bigram tuples, a bigram the two sentences share is automatically **one** node. That shared node is where a walk can cross from one sentence to the other.
4. Generate a new sentence with a random START-to-END walk. The walk is a depth-first search that shuffles each node's successors and never revisits a node, so it can't loop and always finds a path when one exists. The path is turned back into words by taking the first bigram whole and then the second word of every later bigram.

**Departures from the spec, and why:**

- The spec's pseudo-code gives the last bigram an edge to END but no incoming edge from the previous bigram, which would break the chain. Here every bigram is linked from its predecessor, and the last one also links to END.
- A shared bigram at the **end** of one sentence creates a shortcut. For example, "heavy rain caused …" plus "… after hours of heavy rain" allows the path START → *heavy rain* → END. Paths shorter than the shorter input sentence are therefore rejected and re-sampled, up to 25 attempts, after which S1 is used.

## 7. Ordering clusters

Each cluster's output is ordered by the document position of its S1, so the summary follows the flow of the source text.

## Reproducibility

Every random choice — centroid initialisation, tie-breaks and graph walks — draws from a `random.Random(seed)` instance set by `--seed`. Nothing uses the global random state.
