# Problem statement

Build a summarizer for a plain-text document. The project has two parts.

## Part 1: Ranking sentences with PageRank

1. **Preprocessing.** Split the document into sentences, one per line, and keep them in their original order. For each sentence:
   - remove punctuation;
   - lowercase it;
   - tokenize it into words;
   - remove English stopwords;
   - lemmatize each word with WordNet.
2. **TF-IDF representation.** For word *w* and sentence *s*:
   - TF(w, s) = number of times w occurs in s
   - IDF(w) = log(N / number of sentences that contain w), where N is the total number of sentences
   - TF-IDF(w, s) = TF(w, s) × IDF(w)

   Build a word × sentence matrix whose columns are the sentence vectors.
3. **PageRank.** Build a complete, undirected, weighted graph with one node per sentence. Each edge's weight is the cosine similarity between the two sentences' TF-IDF vectors. Run PageRank and output the top-n sentences in their original document order (`Summary_PR.txt`).

## Part 2: Reducing redundancy

The top PageRank sentences often repeat each other. Two remedies are required.

### A. Maximal Marginal Relevance

Re-rank sentences with MMR:

```
MMR = argmax over D_i in R \ S of [ λ·Imp(D_i) − (1 − λ)·max over D_j in S of Sim(D_i, D_j) ]
```

Imp is the PageRank score and Sim is the similarity between sentences. Output the top-n sentences in document order (`Summary_MMR.txt`).

### B. Clustering and word-graph fusion

1. **K-means.** Cluster the TF-IDF vectors:
   - pick K;
   - choose K starting centroids with `random.sample`;
   - assign each sentence to its most cosine-similar centroid;
   - set each centroid to the mean of its cluster;
   - repeat until the assignments stop changing.
2. **Sentence graph.** For each cluster:
   - S1 is the sentence closest to the centroid; ties are broken at random.
   - S2 is a sentence in the same cluster that shares at least 3 bigrams with S1. If there is no such sentence, the cluster contributes S1 unchanged.
   - Otherwise, build a directed graph from START through each sentence's bigrams to END, with shared bigrams forming a single node.
   - Generate a new sentence by following a random path from START to END.
3. **Ordering.** Order the clusters by the position of their S1 in the original document, and output the generated sentences in that order (`Summary_SentenceGraph.txt`).
