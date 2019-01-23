"""The basic infrastructure for creating a generated file.

Creating a generated file requires understanding dependencies and
scheduling tasks.
"""
from __future__ import absolute_import

import logging

from . import build as kake_lib_build
from . import filemod_db as kake_lib_filemod_db

# Import all the compile modules here.  They register themselves.
# (We could automate this, but I find that more obfuscating than helpful.)
try:
    import all_kake_rules
    all_kake_rules.all_compile_rules()       # Does all the imports
except ImportError:
    logging.warning(
        "`make.py` won't do anything until you add the compile rules!")


# Convenience alias for other modules, like build/kake/server_client.py.
BadRequestFailure = kake_lib_build.BadRequestFailure
CompileFailure = kake_lib_build.CompileFailure


def build_many(outfile_names_and_contexts, num_processes=1, force=False,
               checkpoint_interval=60 * 5):
    """Create, or update if it's out of date, the given filename.

    Raises an CompileFailure exception if it could not create or
    update the file.

    Arguments:
       outfile_name_and_contexts: filenames relative to ka-root, each
          context is an arbitrary dict passed to the build rule for
          outfile_name and all its dependencies.
       num_processes: the number of sub-processes to spawn to do the
          building.  (1 means to build in the main process only.)
       force: if True, force deps to be rebuilt even if they are up-to-date.
       checkpoint_interval: if not None, flush the filemod db every
          this many seconds, approximately.  This records the work
          we've done so we don't have to re-do it if the process dies.

    Returns:
        The list of outfile_names that were actually rebuilt
        (because they weren't up to date).
    """
    # We don't trust that files haven't changed since the last time we
    # were called, so we can't use the mtime cache.
    kake_lib_filemod_db.clear_mtime_cache()
    return kake_lib_build.build_with_optional_checkpoints(
        outfile_names_and_contexts, num_processes, force, checkpoint_interval)


def build(outfile_name, context={}, num_processes=1, force=False,
          checkpoint_interval=60 * 5):
    """Create, or update if it's out of date, the given filename.

    Raises an CompileFailure exception if it could not create or
    update the file.

    Arguments:
       outfile_name: filename relative to ka-root.
       context: an arbitrary dict passed to all build rules for this file.
       num_processes: the number of sub-processes to spawn to do the
          building.  (1 means to build in the main process only.)
       force: if True, force deps to be rebuilt even if they are up-to-date.
       checkpoint_interval: if not None, flush the filemod db every
          this many seconds, approximately.  This records the work
          we've done so we don't have to re-do it if the process dies.

    Returns:
        [outfile_name] if it had to be rebuilt (because it wasn't up
        to date), or [] else.
    """
    return build_many([(outfile_name, context)], num_processes, force,
                      checkpoint_interval)
