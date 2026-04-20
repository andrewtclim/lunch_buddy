"""
conftest.py -- runs before every test module.

Patches out Google / Vertex AI libraries so recommend.py can be
imported without credentials or a network connection.  All patched
names return MagicMock objects that swallow attribute access and
calls silently.
"""

import sys
from unittest.mock import MagicMock

# ── stub out every google / vertex module that recommend.py touches ──
for mod in [
    "google.cloud.aiplatform",       # aiplatform.init()
    "vertexai",                      # vertexai.init()
    "vertexai.generative_models",    # GenerativeModel, GenerationConfig
    "google.genai",                  # genai.Client()
    "google.genai.types",            # types.EmbedContentConfig
]:
    sys.modules[mod] = MagicMock()

# ── make models/gemini_flash_rag importable as a plain package ──
import os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "models", "gemini_flash_rag"),
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "fastapi"),
)
