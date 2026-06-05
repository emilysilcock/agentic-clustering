import sys as _sys

if _sys.platform == "win32":
    # AVG (and similar AV) MITM-intercepts TLS with a non-public root that lives in the
    # Windows cert store but is absent from certifi. truststore routes Python's TLS through
    # the OS trust store so requests/httpx/etc. trust it. Must happen before any HTTPS call.
    import truststore as _truststore

    _truststore.inject_into_ssl()

    # Windows default stdout/stderr encoding is cp1252, which crashes any
    # `print()` of LLM-generated text containing common chars outside that
    # codec (smart quotes, em/en dashes, non-breaking hyphen). Observed in
    # the TopicGPT refinement phase 2 (`print(RenderTree(...))`). Force
    # utf-8 here so every entry point under `benchmarking/` is safe.
    for _stream in (_sys.stdout, _sys.stderr):
        _reconfigure = getattr(_stream, "reconfigure", None)
        if _reconfigure is not None:
            _reconfigure(encoding="utf-8", errors="replace")
