# Clinical Named Entity Recognition (NER) with Spark NLP

This repository contains a custom Named Entity Recognition (NER) pipeline for clinical text data using Spark NLP for Healthcare. The pipeline leverages multiple specialized pre-trained models and creates a unified model with intelligent entity prioritization.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Entity Prioritization](#entity-prioritization)
- [Training Process](#training-process)
- [Performance Metrics](#performance-metrics)
- [Error Analysis](#error-analysis)
- [Interactive Visualization](#interactive-visualization)
- [Use Cases](#use-cases)
- [References](#references)

## Overview

This project implements a Named Entity Recognition (NER) system for clinical text data using PySpark and John Snow Labs' Spark NLP libraries. The system combines three specialized pre-trained medical NER models to create a comprehensive entity recognition system that can identify clinical entities, personal identifiable information, and medication details in medical texts.

The key innovation is the entity prioritization system that intelligently resolves conflicts when multiple models identify the same token, ensuring that the most specific and appropriate entity type is assigned. The NER predictions are then converted to CoNLL format and used to train a unified custom NER model.

## Requirements

- Python 3.7+
- PySpark 3.4.1
- Spark NLP 6.0.0
- Spark NLP for Healthcare (JSL) 
- License key for JSL library
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow 1.15.0 (for model training)

## Installation

1. Setup a Python environment:
```bash
conda create -n spark_nlp_env python=3.7
conda activate spark_nlp_env
```

2. Install PySpark and Spark NLP:
```bash
pip install --upgrade pyspark==3.4.1 spark-nlp==6.0.0
```

3. Install Spark NLP for Healthcare:
```bash
pip install spark-nlp-jsl --extra-index-url https://pypi.johnsnowlabs.com/$SECRET
```

4. Install visualization dependencies:
```bash
pip install matplotlib seaborn scikit-learn pandas plotly kaleido
```

5. Install TensorFlow for model training:
```bash
pip install tensorflow==1.15.0 numpy==1.16.4 tensorflow-addons
```

6. Set up your license key:
   - Create a `license.json` file with your John Snow Labs license key
   - Place it in your working directory or in your Google Drive if using Colab

## Dataset

The project utilizes the `mtsamples_classifier.csv` dataset, which contains a collection of clinical text examples across various medical specialties. This dataset provides a diverse range of clinical notes, ideal for training and evaluating NER models on medical text.

If you don't have access to the mtsamples dataset, you can use other publicly available healthcare datasets like:
- MIMIC-III (requires credentialing)
- i2b2 challenge datasets
- Medical transcription samples

## Pipeline Architecture

The NLP pipeline consists of the following components:

1. **Document Assembly**: Converts raw text into document objects
2. **Sentence Detection**: Segments text into sentences
3. **Tokenization**: Splits sentences into tokens
4. **Word Embeddings**: Uses pre-trained clinical word embeddings
5. **Named Entity Recognition**: Applies three different specialized models:
   - `ner_clinical`: Identifies medical conditions, procedures, and tests
   - `ner_deid_generic_augmented`: Recognizes personal identifiable information
   - `ner_posology`: Extracts medication and dosage information
6. **NER Converters**: Transforms predictions into chunks for analysis

### Pipeline Implementation

```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("clinical_ner")

deid_ner = MedicalNerModel.pretrained("ner_deid_generic_augmented", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("deid_ner")

posology_ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("posology_ner")

ner_pipeline = Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    word_embeddings,
    clinical_ner,
    deid_ner,
    posology_ner,
    clinical_converter,
    deid_converter,
    posology_converter
])
```

## Entity Prioritization

The system uses a priority-based approach to resolve conflicts when multiple models identify the same token:

1. **Highest Priority**: Posology entities (DRUG, DOSAGE, FREQUENCY, DURATION)
2. **Medium Priority**: De-identification entities (NAME, DATE, LOCATION, DOCTOR, HOSPITAL)
3. **Lowest Priority**: Clinical entities (PROBLEM, TREATMENT, TEST)

This prioritization ensures that the most specific entity type is selected. For example, "Aspirin" would be labeled as DRUG (from posology_ner) rather than TREATMENT (from clinical_ner).

```python
# Priority-based entity selection
if row['posology_label'] != 'O':
    merged_label = row['posology_label']
elif row['deid_label'] != 'O':
    merged_label = row['deid_label']
elif row['clinical_label'] != 'O':
    merged_label = row['clinical_label']
else:
    merged_label = 'O'
```

### Why Merge Multiple Models?

We use three specialized NER models because each focuses on different aspects of clinical text:

1. **Clinical NER** identifies medical conditions and treatments
2. **DeID NER** recognizes personal information that might require anonymization
3. **Posology NER** specializes in medications and dosage information

By merging them with intelligent prioritization, we create a comprehensive system that provides the most precise entity labels. For example, "Aspirin 10mg daily" will have "Aspirin" labeled as DRUG (rather than TREATMENT), "10mg" as DOSAGE, and "daily" as FREQUENCY.

## Training Process

The NER model training process consists of:

1. **CoNLL Conversion**: Converting merged NER predictions to CoNLL format
2. **Dataset Splitting**: Dividing into training (80%) and testing (20%) sets
3. **Model Configuration**: Setting up MedicalNerApproach with:
   - 10 maximum epochs
   - 0.003 learning rate
   - Early stopping with patience=3
   - Batch size of 8
4. **Training Execution**: Training on the training set with validation split
5. **Model Evaluation**: Performance assessment using precision, recall, and F1

```python
nerTagger = MedicalNerApproach()\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setLabelColumn("label")\
    .setOutputCol("ner")\
    .setMaxEpochs(10)\
    .setLr(0.003)\
    .setBatchSize(8)\
    .setRandomSeed(42)\
    .setVerbose(1)\
    .setEvaluationLogExtended(True)\
    .setEnableOutputLogs(True)\
    .setIncludeConfidence(True)\
    .setValidationSplit(0.2)\
    .setUseBestModel(True)\
    .setEarlyStoppingCriterion(0.04)\
    .setEarlyStoppingPatience(3)
```

The training progress shows steady improvement in both loss reduction and performance metrics:

![Training Progress](images/A.jpg)

The entity-specific F1 scores demonstrate how different entity types learn at varying rates:

![Entity-Specific F1 Scores](images/B.jpg)

## Performance Metrics

The trained model achieves strong performance across entity types as seen in the detailed metrics table:

![Detailed Training Metrics](images/C.jpg)

A visual representation of F1 scores by entity type shows the relative performance:

![F1 Scores by Entity Type](images/entity_f1_scores.png)

The precision, recall, and F1 scores for each entity type demonstrate the model's effectiveness:

![Precision, Recall, and F1 by Entity Type](images/entity_metrics_comparison.png)

The confusion matrix shows excellent performance with most entities correctly classified:

![Confusion Matrix](images/confusion_matrix_mpl.png)

## Error Analysis

The error analysis reveals strong overall performance with 85% correct predictions:

![Error Distribution](images/error_distribution.png)

Error breakdown:
- Correct predictions: 85.0%
- Wrong entity type: 6.0%
- False negatives: 5.5%
- False positives: 3.5%

A pie chart visualization provides an alternative view of error distribution:

![Error Type Distribution](images/error_distribution_pie.png)

Entity-specific findings:
- DATE and NAME entities have the highest F1 scores (>90%)
- DOSAGE entities show slower improvement due to format variations
- DRUG and PROBLEM entities show steady improvement during training

The comparison before and after fine-tuning shows notable improvements:

![F1 Score Improvement](images/model_improvement.png)

## Interactive Visualization

The model includes an interactive visualization component for easy interpretation of predictions:

![Interactive NER Visualization](images/D.jpg)

This visualization shows how the model identifies different entity types in clinical text, with color-coding for each entity category. Interactive HTML versions are available in the repository:
- `confusion_matrix_interactive.html`
- `entity_metrics_interactive.html`
- `error_distribution_interactive.html`
- `performance_comparison_interactive.html`

## Use Cases

This NER system can be used for:

1. **Clinical Information Extraction**: Automatically extracting medical conditions, treatments, and tests from clinical notes
2. **De-identification**: Identifying and anonymizing personal health information for data sharing
3. **Medication Analysis**: Extracting medication details including drugs, dosages, and frequencies
4. **Clinical Research**: Analyzing large volumes of medical text for research purposes
5. **Healthcare Documentation**: Enhancing medical record systems with automated entity recognition

## References

- Spark NLP for Healthcare Documentation: [John Snow Labs](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators)
- Medical NER Models: [Spark NLP Models Hub](https://nlp.johnsnowlabs.com/models)
- CoNLL Format: [CoNLL-2003 Format Description](https://www.clips.uantwerpen.be/conll2003/ner/)
- Medical Text Samples Dataset: [MTSamples](https://www.mtsamples.com/)

---

© 2025 John Snow Labs | Spark NLP for Healthcare
