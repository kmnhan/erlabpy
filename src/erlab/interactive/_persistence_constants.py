"""Names and versions shared by interactive persistence code.

Keep this module free of imports so tools and workspace readers can share the
saved-file contract without importing each other's implementation.
"""

# Data variables, coordinates, and encoded attributes.
ITOOL_DATA_NAME: str = "<erlab-itool-data>"
SAVED_TOOL_DATA_NAME: str = "<saved-tool-data>"
TOOL_DATA_BLOB_NAME_ATTR: str = "tool_data_blob_name"
SAVED_TOOL_DATA_REFERENCE_DIM: str = "<saved-tool-data-reference>"
SAVED_TOOL_DATA_BLOB_DIM_PREFIX: str = "<saved-tool-data-blob-"
TOOL_ATTRS_VERSION_ATTR = "_erlab_tool_attrs_version"
TOOL_ENCODED_ATTRS_ATTR = "_erlab_tool_encoded_attrs"
TOOL_ATTRS_VERSION = 1
TOOL_ATTR_TRANSPORT_KEYS = frozenset((TOOL_ATTRS_VERSION_ATTR, TOOL_ENCODED_ATTRS_ATTR))

# Tool input and provenance metadata.
TOOL_SOURCE_SPEC_ATTR = "tool_source_spec"
TOOL_SOURCE_BINDING_ATTR = "tool_source_binding"
TOOL_SOURCE_STATE_ATTR = "tool_source_state"
TOOL_SOURCE_AUTO_UPDATE_ATTR = "tool_source_auto_update"
TOOL_INPUT_PROVENANCE_SPEC_ATTR = "tool_input_provenance_spec"
TOOL_SCRIPT_INPUTS_ATTR = "tool_script_inputs"
TOOL_PRIMARY_INPUT_ATTR = "tool_primary_input"
TOOL_DATA_REFERENCES_ATTR = "tool_data_references"
NONE_TOOL_DATA_NAME = "<none-value>"

# ImageTool and Manager provenance metadata.
MANAGER_LIVE_SOURCE_SPEC_ATTR = "manager_node_live_source_spec"
MANAGER_PROVENANCE_SPEC_ATTR = "manager_node_provenance_spec"
ITOOL_PROVENANCE_SPEC_ATTR = "itool_provenance_spec"

# Opaque executable-payload metadata.
CODE_PAYLOAD_ENTRIES_ATTR = "erlab_code_trust_payload_entries"

# Workspace document formats and legacy transactions.
WORKSPACE_SCHEMA_VERSION = 6
WORKSPACE_LEGACY_SCHEMA_VERSION = 3
WORKSPACE_MANIFEST_ATTR = "imagetool_workspace_manifest"
WORKSPACE_LEGACY_TEMP_GROUP_PREFIXES = ("__itws_pending_", "__itws_backup_")
WORKSPACE_TRANSACTION_GROUP_PREFIX = "__itws_txn_"
WORKSPACE_ENCODED_ATTRS_ATTR = "_erlab_workspace_encoded_attrs"
WORKSPACE_ENCODED_ATTRS_VERSION = 1
WORKSPACE_REPLAY_SOURCE_BLOB_NAME = "<manager-replay-source-data>"

# Immutable workspace storage.
WORKSPACE_OBJECTS_GROUP = "__itws_objects"
WORKSPACE_STAGING_GROUP = "__itws_staging"
WORKSPACE_GENERATIONS_GROUP = "__itws_generations"
WORKSPACE_GENERATION_WIDTH = 20
WORKSPACE_ID_ATTR = "imagetool_workspace_id"
