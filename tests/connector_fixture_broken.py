"""A connector plugin that fails on import, which is the most common way.

Separate module on purpose: a module-level raise takes the whole module
with it, so the healthy fixtures could not live here. A missing
transitive dependency behaves exactly like this.
"""

raise RuntimeError("this connector plugin is broken on purpose")
