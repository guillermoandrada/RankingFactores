# Profile Editor Architecture

## Tree Navigator + Node Editor

- **Left pane (35%)**: Compact tree of the scoring profile. Each node shows name, method, weight-from-parent, and a ⚠ badge if validation fails.
- **Right pane (65%)**: Editor for the **selected node only**. No recursive expanders.

## Selection & State

- `st.session_state["profile_editor_selected_node_id"]` – Currently selected node. Clicking a tree node sets this and triggers a rerun.
- `st.session_state[f"profile_editor_store_{profile_name}"]` – In-memory `ProfileStore` instance. Persists across reruns.
- Profile-level fields (normalization, winsorization) are stored in session state keys per profile.

## Flat Store

- `nodes: Dict[node_id, Node]` where `Node = {id, type, name, method, params, children}`.
- `children`: `[{child_id, weight, enabled}]` – `child_id` is either a node_id (subfactor) or metric name (leaf).
- Migration from legacy `{name: {inputs: {child: weight}}}` is done in `migrate_legacy_to_flat()`.
- Export back to API payload via `flat_to_export_payload()`.

## Module Layout

- `profile_store.py` – Flat store, CRUD, normalize/equalize/sort, duplicate subtree, delete node.
- `validators.py` – Global validation (weight sum, empty nodes). Returns list of `{node_id, kind, message}`.
- `tree_nav.py` – Renders tree, handles selection, quick actions (Add metric, Add subfactor, Duplicate, Delete).
- `node_editor.py` – Node path, name/method fields, components `st.data_editor` table, weight actions.
- `previews.py` – Graph (st.graphviz_chart) and JSON tabs.

