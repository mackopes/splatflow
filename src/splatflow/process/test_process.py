import pathlib
import sys
import time

from splatflow.tabs.datasets.dataset import Dataset, ProcessedDataset


def process_dataset(dataset_dir: pathlib.Path):
    print(f"starting to process {dataset_dir}")
    delay = 15

    for i in range(delay):
        time.sleep(1)
        print(f"processing {100 * (i + 1) / delay}%")

    dataset = Dataset.load_from_directory(dataset_dir)
    processed_dataset = ProcessedDataset(
        f"test_dataset_{len(dataset.processed_datasets)}",
        {"test_setting_a": 2, "haha": "macka"},
    )
    dataset.add_processed_dataset(processed_dataset)
    print("Processing completed!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m splatflow.process.test_process <dataset_dir>")
        sys.exit(1)

    dataset_dir = pathlib.Path(sys.argv[1])
    process_dataset(dataset_dir)
