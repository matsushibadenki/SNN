# snn_research/hardware/__init__.py
# (更新)

from .profiles import get_hardware_profile, loihi_profile
# --- ▼ 修正: Exporter を Compiler に変更 ▼ ---
from .compiler import NeuromorphicCompiler

__all__ = [
    "get_hardware_profile", 
    "loihi_profile",
    "NeuromorphicCompiler" # Exporter を Compiler に変更
]
# --- ▲ 修正 ▲ ---
