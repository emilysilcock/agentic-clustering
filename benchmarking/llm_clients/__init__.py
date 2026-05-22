"""Local LLM clients (subprocess wrappers and SDK adapters).

Sibling modules:
- ``claude_code``: subprocess wrapper around the local Claude Code CLI for
  Opus 4.7 calls billed against the Max plan rather than the API.

Future siblings will hold the Anthropic Batch API client (Haiku via SDK) for
Viswanathan+, TopicGPT, and Huang & He per SPEC §5.6.2.
"""
