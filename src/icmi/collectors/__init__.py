"""Collectors: per-platform raw data collection.

Each collector fetches listings from one platform, preserves raw payloads
verbatim, and produces records in the shared raw schema plus a run manifest.
Collectors are mode-agnostic: the same code runs locally or on GitHub Actions
and must produce identical output formats in both modes.
"""
