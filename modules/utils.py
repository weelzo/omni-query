def reciprocal_rank_fusion(text_results, image_results, k=60):
    scores = {}
    for i, (text_res, image_res) in enumerate(zip(text_results, image_results)):
        for rank, doc_id in enumerate(text_res):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)
        for rank, doc_id in enumerate(image_res):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)