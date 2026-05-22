import sys as _sys

if _sys.platform == "win32":
    # AVG (and similar AV) MITM-intercepts TLS with a non-public root that lives in the
    # Windows cert store but is absent from certifi. truststore routes Python's TLS through
    # the OS trust store so requests/httpx/etc. trust it. Must happen before any HTTPS call.
    import truststore as _truststore

    _truststore.inject_into_ssl()
