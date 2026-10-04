"""Code that runs *inside* the sandbox child: the program-facing API, the static checker and
the worker loop. This package must stay free of heavy imports (numpy, cv2, pydantic): the
child applies a small ``RLIMIT_AS`` and anything that maps large address space would make
every program die with ``MemoryError``.
"""
