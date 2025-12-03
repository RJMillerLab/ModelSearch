# External Dependencies

This directory contains minimal necessary code from external repositories to avoid requiring separate repo installations.

## Structure

- `tab2know/`: Minimal Tab2Know code for table classification
  - `run_inference.py`: Main inference script
  - `tab2know/`: Core tab2know modules
  - Note: Models directory is not included (uses external repo if available)

- `dialite/alite/`: Minimal ALITE code for table integration
  - `alite_fd.py`: Main ALITE FD algorithm
  - Supporting modules for graph algorithms

## Usage

The code automatically detects these local copies first, then falls back to external repositories if:
1. Local `others/` directory exists and has required files
2. Environment variables are set (e.g., `TAB2KNOW_REPO`, `DIALITE_INTERNAL_REPO`)
3. External repos in standard locations

This allows the system to work standalone while still supporting external repo configurations.

