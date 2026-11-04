============
Contributing
============

Development setup
=================

.. code-block:: bash

   python -m venv .venv
   .venv/bin/pip install -e '.[dev]'
   .venv/bin/ruff check src tools tests examples benchmarks
   .venv/bin/ruff format --check src tests examples benchmarks
   .venv/bin/python -m pytest -q

The full test suite runs on CPU in well under a few minutes. Tests marked
``slow`` perform tiny optimisation checks and can be skipped with
``-m 'not slow'``.

Ground rules
============

* Keep the implementation in importable modules under ``src/eventflow``.
  Notebooks and scratch scripts are not part of the library.
* Every numerical routine needs a test against an analytic or hand-computed
  value, not only a shape check.
* Never let a model see the event it is predicting. New inputs to the context
  encoder must come with a causality test (see ``tests/unit/test_causality.py``).
* Metrics declare the dataset capabilities they require. Do not compute a
  statistic from a field the data source does not genuinely provide.
* Do not commit datasets, checkpoints, run outputs, credentials or Markdown
  files. Small synthetic fixtures under ``tests/fixtures/`` are the exception.
* Commit messages follow ``<type>: <description>`` with types ``feat``,
  ``fix``, ``refactor``, ``docs``, ``test``, ``chore``, ``perf`` or ``ci``.

Reporting results
=================

Results must state the configuration, dataset fingerprint, seeds and code
revision that produced them (the run manifest records all four). Do not
report numbers from runs whose manifest is missing.
