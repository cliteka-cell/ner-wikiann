# Named Entity Recognition with DistilBERT

🔗 **[Try the live demo](https://huggingface.co/spaces/Reshiman/ner-wikiann)**

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

**Domain mismatch:** WikiANN is built from Wikipedia — formal, encyclopedic text
about well-known entities. The model performs confidently on Wikipedia-style sentences
(e.g. *"Barack Obama served as the 44th President of the United States"*) but struggles
with casual, conversational text (e.g. *"Jake Johnson loves Sarah who lives in Michigan LA in USA"*).
"Sarah" gets misclassified as ORG and "LA" as ORG — both with low confidence scores,
which correctly signals the model's own uncertainty. This is a known limitation of
training on domain-specific corpora and a strong argument for diverse training data.

## Error Analysis

Ran the model against the full test set (80,326 tokens) and categorized every mistake.

### Error Breakdown

| Error Type | Count | % of Errors | Meaning |
|------------|-------|-------------|---------|
| Wrong entity type | 4,211 | 74.2% | Found the entity, classified it wrong |
| False negative | 769 | 13.5% | Missed a real entity entirely |
| False positive | 698 | 12.3% | Predicted an entity that isn't one |

The model is good at *finding* entities but struggles to *classify* them correctly.
Low false negatives (13.5%) confirm this — missing entities is rare.

### Top Confusion Pairs

| True Label | Predicted | Count | Reason |
|------------|-----------|-------|--------|
| I-LOC | I-ORG | 685 | Geopolitical ambiguity |
| I-ORG | I-LOC | 594 | Geopolitical ambiguity |
| I-PER | I-ORG | 515 | Organizations named after people |
| I-ORG | I-PER | 506 | Organizations named after people |
| B-ORG | B-LOC | 402 | Location/organization boundary |

**LOC ↔ ORG accounts for 1,279 wrong-type errors** — the single biggest failure mode.
In Wikipedia text, geopolitical entities like countries and cities frequently act as
organizations (*"France announced..."*), creating systematic ambiguity that no model
can resolve without broader context.

### Real Examples from the Test Set

**I-LOC → I-ORG (model predicts ORG, truth is LOC)**
- *"Encash Network Service"* — "Network Service" reads naturally as an organization.
  The model's prediction is the more intuitive call.

**I-PER → I-ORG (model predicts ORG, truth is PER)**
- *"List of The O.C. characters"* — ground truth tags "The O.C." as PER because it
  references fictional people. The model predicts ORG, which is arguably more correct
  for a TV show title. This is a dataset annotation error, not a model failure.

**B-ORG → B-LOC (model predicts LOC, truth is ORG)**
- *"Jhansi (JHS)"* — Jhansi is a city in India, tagged ORG in the dataset but
  predicted LOC by the model. The model is correct.
- *"Mittani"* — an ancient empire that simultaneously functions as a civilization,
  location, and political entity. No annotation guideline handles this cleanly.

### Key Finding

In several cases the model's predictions are **more linguistically defensible than
the ground truth labels**. This points to annotation inconsistencies in WikiANN itself —
a known limitation of automatically constructed NER datasets derived from Wikipedia
infoboxes. Manual annotation with explicit guidelines would likely improve both
dataset quality and model performance.

## How to Run

Open `exploration.ipynb` in Google Colab and run all cells.
No local setup required — all dependencies are installed in the first cell.

## Tech Stack

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [HuggingFace Datasets](https://github.com/huggingface/datasets)
- Python 3.x, PyTorch

## Live Demo

Try it yourself: [huggingface.co/spaces/Reshiman/ner-wikiann](https://huggingface.co/spaces/Reshiman/ner-wikiann)
