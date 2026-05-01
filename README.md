# Named Entity Recognition with DistilBERT

Fine-tuned DistilBERT on the WikiANN English dataset for Named Entity Recognition (NER).
Trained to detect **people, organizations, and locations** in plain text.

## Results

| Metric   | Score  |
|----------|--------|
| F1       | 0.8202 |
| Accuracy | 0.9291 |
| Loss     | 0.2522 |

### Training Progress

| Epoch | Training Loss | Validation Loss | F1     | Accuracy |
|-------|--------------|-----------------|--------|----------|
| 1     | 0.3028       | 0.2603          | 0.7980 | 0.9220   |
| 2     | 0.2134       | 0.2467          | 0.8151 | 0.9280   |
| 3     | 0.1596       | 0.2522          | 0.8202 | 0.9291   |

Note: validation loss increased slightly at epoch 3 while F1 continued improving —
a mild sign of overfitting. A next step would be early stopping or adding dropout.

## Dataset

[WikiANN (English)](https://huggingface.co/datasets/wikiann) — 20,000 training sentences
sourced from Wikipedia, annotated with PER, ORG, and LOC entity types.

### Class Distribution

| Label | Count  | %     |
|-------|--------|-------|
| O     | 81,362 | 50.7% |
| B-PER | 9,164  | 5.7%  |
| I-PER | 14,698 | 9.2%  |
| B-ORG | 9,422  | 5.9%  |
| I-ORG | 23,226 | 14.5% |
| B-LOC | 9,345  | 5.8%  |
| I-LOC | 13,177 | 8.2%  |

`O` tokens make up 50.7% of all labels — making accuracy a misleading metric.
F1 is the more honest measure of NER performance.

## Model

- **Base model:** `distilbert-base-uncased`
- **Task:** Token classification (NER)
- **Epochs:** 3
- **Learning rate:** 2e-5
- **Batch size:** 16

## Example Outputs

Input: *"Abraham Lincoln was born in Larue County, KY and Lincoln Memorial in
Washington D.C. was built for him by Henry Bacon."*

| Token            | Entity | Confidence |
|------------------|--------|------------|
| Abraham Lincoln  | PER    | 0.76       |
| Larue County     | LOC    | 0.95       |
| KY               | LOC    | 0.50       |
| Lincoln Memorial | ORG    | 0.94       |
| Washington D.C.  | LOC    | 0.75       |
| Henry Bacon      | PER    | 0.91       |

Input: *"Elon Musk founded SpaceX in California and Tesla in Texas."*

| Token      | Entity | Confidence |
|------------|--------|------------|
| Elon Musk  | PER    | 0.97       |
| SpaceX     | ORG    | 0.95       |
| California | LOC    | 0.99       |
| Tesla      | ORG    | 0.85       |
| Texas      | LOC    | 0.99       |

## Linguistic Observations

**Lexical ambiguity across spans:** "Lincoln" appears twice in the sentence — once as
part of a person name and once as part of a landmark. The model assigns lower confidence
to "Abraham Lincoln" (0.76) likely because the repeated token creates ambiguity in context.

**Abbreviations hurt confidence:** "KY" (Kentucky) receives only 0.50 confidence.
Abbreviations are underrepresented in Wikipedia-sourced training data, so the model
struggles with abbreviated location names.

**Punctuation fragments tokens:** "Washington D.C." receives 0.75 confidence —
the periods in "D.C." cause WordPiece tokenization to split the token in unexpected ways,
reducing the model's certainty.

**Annotation guideline ambiguity:** "Lincoln Memorial" is tagged as ORG (0.94) rather
than LOC. This reflects a genuine annotation challenge — landmarks and institutions sit
on the boundary between organization and location, and the correct label often depends
on annotation guidelines rather than linguistic facts.

**Subword tokenization on proper nouns:** DistilBERT lowercases all input and uses
WordPiece tokenization. Names like "Elon" get split into `el` + `##on`, which can
fragment entity spans and reduce confidence on rare or unusual names.

**Mild overfitting at epoch 3:** Validation loss increased from 0.2467 to 0.2522
between epochs 2 and 3 while F1 kept improving. This suggests the model began
memorizing training examples. Early stopping or increased dropout would likely
improve generalization.

## How to Run

Open `exploration.ipynb` in Google Colab and run all cells.
No local setup required — all dependencies are installed in the first cell.

## Tech Stack

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [HuggingFace Datasets](https://github.com/huggingface/datasets)
- Python 3.x, PyTorch
