"""
Latent control modules for image-token generation.

Design goals:
- Fixed-length memory/control tokens to avoid context blow-up
- No KV rollback (prefix injection updates KV cache in-place)
- Minimal dependencies: PyTorch only
"""


