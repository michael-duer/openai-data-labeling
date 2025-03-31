# Relationship labeling using OpenAI API

This repository contains scripts which use the OpenAI API to label relationships between two named entities within sentences. The labeled output is saved as CSV files, which can be used to train machine learning models for relationship detection.

## Features

- **Batch Processing:** Efficiently processes sentences in batches.
- **Relationship Categorization:** Detect and categorizes relationships using predefined categories:
  - Positive1 (one-sided), Positive2 (mutual)
  - Neutral1 (one-sided), Neutral2 (mutual)
  - Negative1 (one-sided), Negative2 (mutual)
  - None (no relationship detected)
- **CSV Output:** Exports the labeled data in CSV format.
- **Evaluation Metrics:** Generates confusion matrices and calculates key performance metrics (accuracy, Krippendorff’s Alpha, and Brennan-Prediger’s Alpha) to assess model performance.

## Project Structure

```plaintext
├── data
│   ├── api_output/               # OpenAI-labeled output files
│   ├── evaluation/
│   │   ├── confusion_matrices/   # Confusion matrices for analysis
│   │   └── evaluation.csv        # Evaluation metrics/results
│   ├── samples/                  # Sampled input sentences for labeling
│   └── TrainingData.csv          # Ground truth labeled data
│
├── prompts/                      # Prompt templates used with the API
│
├── scripts/
│   ├── evaluate_output.R         # Evaluates model output vs ground truth
│   ├── generate_samples.R        # Create samples from labeled data
│   ├── helper_functions.py       # Utility functions
│   └── run.py                    # Main script to run API labeling
│
├── .env                          # Environment variables (OpenAI API key)
├── .gitignore                    # Files to ignore in version control
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Requirements

- **Python:** Version 3.13 or later.
  
  Needed packages are listed in `requirements.txt`.
- **R** (only needed for creating samples and evaluating the output files)
  
  Needed packages are: `dplyr`, `ggplot2`, `gridExtra` and `grid`.
- **OpenAI API Access**
  
  Obtain an API key from the [OpenAI API Platform](https://platform.openai.com/api-keys).

## Setup

1. **Clone the Repository**

```bash
git clone https://github.com/michael-duer/openai-data-labeling.git
cd openai-data-labeling
```

2. **Install Python Dependencies**

```bash
 pip install -r requirements.txt
```

3. **Configure API Key**
   
   Rename the provided `.env.example` file to `.env` and replace the placeholder with your actual OpenAI API key:

```env
OPENAI_API_KEY = your_openai_api_key
```

## Usage

1. **Prepare Input Data**
   
   Place a CSV file containing sentences in the `data/samples/` folder. The file should include the fields: `sentence`, `head`, `tail`, and `relation`. You may also use one of the provided samples.
3. **Configure the Script**
   
   In `scripts/run.py`, adjust the following parameters:

   - **model_id**: The OpenAI model identifier (e.g., `gpt-4-turbo`).
   - **system_prompt_file**: The prompt file (located in the `prompts/` folder).
   - **input_file**: The CSV file name with the input sentences.
   - **Batch Size**: Modify the batch size in the `generate_prompts()` function based on the token limits of your chosen model.

4. **Run the Labeling Process**

```bash
 cd scripts
 python run.py
```

The labeled output will be saved as two files in the `data/api_output/` directory with the filenames formatted as:

```
output_{model}_{prompt}_{sample}.csv
detailed_output_{model}_{prompt}_{sample}.csv
```

## Output Format

The output CSV file includes:

- `sentence`: The original sentence.
- `head`: First named entity.
- `tail`: Second named entity.
- `relation_predicted`: Predicted relationship.
- `relation_true`: True relationship.

The detailed output file additionally also contains the following columns:

- `correct_detailed`: Indicates if the detailed relationship labels match.
- `relation_true_simplified`: Simplified version of the true relationship without one-sided/mutual (e.g. `neutral` instead of `neutral1`).
- `relation_predicted_simplified`: Simplified version of the predicted relationship without one-sided/mutual (e.g. `neutral` instead of `neutral1`).
- `correct_simplified`: Indicates if the simplified relationship labels match.

## Evaluation Metrics

The evaluation file contains:

- **Dataset Details**:
  - Sample file name and size.
  - Model and prompt used.
- **Performance Metrics** (for both detailed and simplified labels):
  - Accuracy
  - Krippendorff’s Alpha
  - Brennan-Prediger’s Alpha

These metrics help compare different models and prompt configurations.
