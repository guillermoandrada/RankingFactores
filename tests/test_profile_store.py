from __future__ import annotations

import pytest

from streamlit_app.components.profile_editor.profile_store import migrate_legacy_to_flat


def test_migrate_legacy_to_flat_rejects_circular_reference() -> None:
    profile = {
        "nodes": {
            "Scoring": {"inputs": {"Reuters": 1.0}},
            "Reuters": {"inputs": {"Reuters": 1.0}},
        }
    }

    with pytest.raises(ValueError, match="Circular profile reference"):
        migrate_legacy_to_flat(profile)
