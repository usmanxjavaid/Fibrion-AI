# Importing weaving here triggers its register_module() call as a
# side effect, once, whenever anything imports this package - so any
# agent can safely call get_process_module("weaving") without ever
# importing weaving.py itself.
from core.schema_registry import weaving  # noqa: F401