"""ClusterLLM baseline (Zhang et al., EMNLP 2023).

Pipeline (per SPEC §5.6.2):
- Phase 0: embed docs with Instructor-large.
- Phase 1: entropy-rank ambiguous points, sample triplets (anchor, A, B).
- Phase 2: judge triplets with Claude Code Opus 4.7 (this is the overnight
  call site — see ``triplet_judge.py``).
- Phase 3: fine-tune Instructor with InfoNCE on the judged triplets (GPU,
  shipped to FASRC).
- Phase 4: re-embed with the fine-tuned encoder, k-means at ``k_in_scope``.

Author source is vendored under ``_vendored/``; only ``predict_triplet.py``
is replaced by our ``triplet_judge.py`` (which uses Claude Code instead of
the legacy openai==0.x API the authors used).
"""
