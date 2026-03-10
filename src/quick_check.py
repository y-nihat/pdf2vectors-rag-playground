from datasets import load_dataset


def main() -> None:
    """Output:
          Built pdf2vectors-rag-playground @ file:///home/anu/projects/llm-consensus/pdf2vectors-rag-playground
        Uninstalled 1 package in 0.75ms
        Installed 1 package in 0.83ms
        README.md: 14.3kB [00:00, 45.5MB/s]
        Resolving data files: 100%|█████████████████████████████████████████████████████████████████████| 1801/1801 [00:00<00:00, 75890.79it/s]
        dict_keys(['json', 'pdf', '__key__', '__url__'])
    """
    dataset = load_dataset("pixparse/pdfa-eng-wds", split="train", streaming=True)
    sample = next(iter(dataset))
    print(sample.keys())  # expected: '__key__', '__url__', 'json', 'ocr', 'pdf', 'tif'


if __name__ == "__main__":
    main()
