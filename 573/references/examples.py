import datasets

elife = datasets.load_dataset("BioLaySumm/BioLaySumm2025-eLife")
"""
DatasetDict({
    train: Dataset({
        features: ['article', 'summary', 'section_headings', 'keywords', 'year', 'title'],
        num_rows: 4346
    })
    validation: Dataset({
        features: ['article', 'summary', 'section_headings', 'keywords', 'year', 'title'],
        num_rows: 241
    })
    test: Dataset({
        features: ['article', 'summary', 'section_headings', 'keywords', 'year', 'title'],
        num_rows: 142
    })
})
"""

plos = datasets.load_dataset("BioLaySumm/BioLaySumm2025-PLOS")