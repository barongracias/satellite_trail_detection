# ASTA (Legacy Implementation)

## Overview

ASTA represents the original baseline implementation used during early
experimentation phases of this project. It is preserved for:

- Reproducibility
- Historical comparison
- Methodological reference

The implementation is not actively maintained and may not conform to the
current modular architecture adopted under `src/`.

## Location

Source file: `src/legacy/ASTA.py`

## Purpose

The ASTA module was used for:

- Initial model prototyping
- Rapid experimentation
- Baseline metric generation

It predates:

- Structured logging
- Decorator-based instrumentation
- Formalised training pipelines
- Unit test coverage

## Reproducibility

To reproduce historical results:

1. Activate the project environment.
2. Ensure dataset structure matches original assumptions.
3. Execute the legacy script directly:

```bash
python src/legacy/ASTA.py