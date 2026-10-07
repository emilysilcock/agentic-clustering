"""ClusterLLM baseline (Zhang et al., EMNLP 2023).

Pipeline:
- Phase 0: embed docs with Instructor-large.
- Phase 1: entropy-rank ambiguous points, sample triplets (anchor, A, B).
- Phase 2: judge triplets with an LLM (the only LLM call site — see
  ``triplet_judge.py``).
- Phase 3: fine-tune Instructor with InfoNCE on the judged triplets (GPU).
- Phase 4: re-embed with the fine-tuned encoder, k-means at ``k_in_scope``.

Author source is vendored under ``_vendored/``; only ``predict_triplet.py``
is replaced by our ``triplet_judge.py`` (gpt-5-mini via the OpenAI Batch API
by default, instead of the legacy openai==0.x API the authors used).
"""
