from summarizer.pipeline import Document, summarize_clusters, summarize_mmr, summarize_pagerank

DOCUMENT = "\n".join([
    "The central bank raised interest rates to fight inflation.",
    "The central bank raised interest rates again this month to fight inflation.",
    "Inflation has stayed high because energy prices keep rising.",
    "The local football team won the championship on Sunday.",
    "Fans celebrated the football team win in the city centre.",
])


def test_pagerank_summary_uses_original_text_in_order():
    document = Document.from_text(DOCUMENT)
    summary = summarize_pagerank(document, 3)
    assert [s.index for s in summary] == sorted(s.index for s in summary)
    assert all(s.original in DOCUMENT.splitlines() for s in summary)


def test_mmr_avoids_picking_both_near_duplicates():
    document = Document.from_text(DOCUMENT)
    picked = {s.index for s in summarize_mmr(document, 2, lambda_=0.5)}
    assert picked != {0, 1}


def test_cluster_summary_is_ordered_by_anchor_position():
    document = Document.from_text(DOCUMENT)
    summary = summarize_clusters(document, k=2, seed=1)
    orders = [item.order for item in summary]
    assert orders == sorted(orders)
    assert 1 <= len(summary) <= 2
