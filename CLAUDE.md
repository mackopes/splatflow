# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Splatflow is a terminal-based tool for managing 3D Gaussian Splatting workflows. It provides a TUI (Terminal User Interface) for running structure-from-motion (COLMAP/GLOMAP) and gsplat training with smart caching - only reruns what changed.

## Development Commands

### Running the Application
```bash
poetry run splatflow
```

### Running Processing/Training Scripts Directly
```bash
# Process a dataset (COLMAP/GLOMAP)
poetry run process --dataset-dir /path/to/dataset --output-dir /path/to/output

# Train a model
poetry run train default --data-dir /path/to/processed --result-dir /path/to/model
```

### Development Tools
```bash
# Type checking
poetry run pyright

# Linting
poetry run ruff check
poetry run ruff format

# Watch mode for tests
poetry run ptw
```

## High-Level Architecture

### TUI Structure

The application is built with Textual and has a 3-tab interface:

1. **Datasets Tab** (`src/splatflow/tabs/datasets/`): Manage raw datasets and processed (colmapped) versions
2. **Models Tab** (`src/splatflow/tabs/models/`): Manage trained Gaussian Splatting models
3. **Queue Tab** (`src/splatflow/tabs/queue/`): Monitor and view logs of running processes

Each tab inherits from `FlowTab` which provides a consistent interface for focus management.

### Data Directory Structure

The application manages data in a hierarchical structure defined by `splatflow_data_root` in `config.toml`:

```
splatflow_data_root/
├── datasets/
│   └── {dataset_name}/
│       ├── input/                    # Raw images
│       ├── splatflow_data.json       # ProcessedDataset metadata
│       ├── {processed_name_1}/       # Colmapped data (different params)
│       └── {processed_name_2}/       # Another processed version
└── models/
    └── {dataset_name}/               # Matches dataset name
        ├── splatflow_data.json       # ProcessedModel metadata
        ├── {model_name_1}/           # Trained model output
        └── {model_name_2}/           # Another trained model
```

### Metadata JSON Files

**For Datasets** (`datasets/{name}/splatflow_data.json`):
- Managed by `SplatflowData` class (list of `ProcessedDataset` objects)
- Each `ProcessedDataset` has: name, state (PENDING/PROCESSING/READY/FAILED), settings
- The `Dataset` class combines computed info (file counts, dates) with JSON data

**For Models** (`models/{dataset_name}/splatflow_data.json`):
- Managed by `SplatflowModelData` class (list of `ProcessedModel` objects)
- Each `ProcessedModel` has: name, dataset_name, state, settings
- No parent class - just direct JSON load/save in workflow

**Key Difference**: Datasets have a complex parent/child relationship with computed metadata. Models are simpler - just JSON persistence.

## Core Workflows

### Processing Workflow (Dataset → ProcessedDataset)

1. User selects dataset and presses 'p' → `action_process_dataset()`
2. `ProcessDialog` modal collects processed dataset name
3. `_process_dataset_worker()` creates `ProcessedDataset` with PENDING state
4. Saves to JSON and adds to queue with state callbacks
5. Queue executes command, callbacks update state to PROCESSING → READY/FAILED
6. UI auto-switches to Queue tab to show progress

### Training Workflow (ProcessedDataset → Model)

1. User selects processed dataset and presses 'm' → `action_train_dataset()`
2. `TrainDialog` modal collects model name and builds `GsplatCommandSettings`
3. `_train_dataset_worker()` creates `ProcessedModel` with PENDING state
4. Saves to `models/{dataset_name}/splatflow_data.json`
5. Adds to queue with state callbacks
6. Queue executes, callbacks update state

### Queue & Subprocess Execution

The queue system (`src/splatflow/main.py`) handles async subprocess execution:

1. **Queue Worker**: Continuous loop checking for pending items (`process_queue_worker()`)
2. **Subprocess Execution**: Uses `asyncio.create_subprocess_exec` with PIPE for stdout/stderr
3. **Real-time Output**: Reads byte-by-byte, handles both `\n` (new line) and `\r` (carriage return for tqdm)
4. **State Callbacks**: Three callbacks update metadata JSON at different stages:
   - `before_exec_callback`: Update state to PROCESSING before execution
   - `on_success_callback`: Update state to READY on success
   - `on_error_callback`: Update state to FAILED on error

### Reactive State Management

- `SplatflowApp.queue` is a reactive property - mutations trigger UI updates
- Use `self.mutate_reactive(SplatflowApp.queue)` after modifying queue items
- Important: This pattern is used to sync subprocess output to the UI in real-time

## Important Implementation Details

### Tree Data Structure in Datasets Tab

The datasets tree has a two-level hierarchy:
- **Parent nodes**: Raw `Dataset` objects (stored as `data=dataset`)
- **Child nodes**: Tuple of `(dataset, processed_dataset)` (stored as `data=(dataset, processed)`)

This allows child nodes to access both the parent dataset and the processed dataset for training workflows.

### Dialog → Settings → Command Pattern

Dialogs return typed command settings objects (not raw strings):
- `ProcessDialog` → returns processed name string
- `TrainDialog` → returns `GsplatCommandSettings` object

The settings objects have:
- `to_dict()`: For JSON serialization (saved to metadata)
- `build()`: For generating subprocess command list

### Subprocess Output Handling

The queue handles tqdm progress bars by:
- Detecting `\r` (carriage return) and overwriting the last line in output array
- Detecting `\n` (newline) and appending new line to output array
- `TQDM_MININTERVAL` is set to limit update frequency

### State Machine for Processing

All long-running tasks follow this state machine:
1. **PENDING**: Created but not started
2. **PROCESSING**: Currently executing
3. **READY**: Completed successfully
4. **FAILED**: Completed with errors

States are persisted to JSON and displayed in UI with visual indicators.

## Code Patterns to Follow

### Adding New Commands to Queue

```python
# 1. Define callbacks to update state
def before_exec_callback():
    # Load metadata, update state to PROCESSING, save

def on_success_callback():
    # Load metadata, update state to READY, save

def on_error_callback():
    # Load metadata, update state to FAILED, save

# 2. Add to queue with callbacks
app.add_to_queue(
    name="Display name for queue",
    command=settings.build(),
    before_exec_callback=before_exec_callback,
    on_success_callback=on_success_callback,
    on_error_callback=on_error_callback,
)

# 3. Switch to queue tab
app.switch_to_queue_tab()
```

### Working with Metadata JSON

```python
# Load metadata
def load_metadata(path: Path) -> SplatflowData:
    if not path.exists():
        return SplatflowData([])
    with open(path, "r") as f:
        data = json.load(f)
    return SplatflowData.from_dict(data)

# Update and save
metadata = load_metadata(path)
metadata.datasets.append(new_item)  # or .models for models
metadata.save(path)
```

## Dependencies & Submodules

- **Hierarchical-Localization (hloc)**: Used for COLMAP/GLOMAP processing (in `submodules/`)
- **gsplat**: 3D Gaussian Splatting implementation
- **Textual**: TUI framework
- **Pyright extraPaths**: Includes hloc submodule for type checking
