# Entity Relation Annotator
_A research tool for labeling and evaluating entity relationships using the OpenAI API_

<ul style="background-color:#FFA0F0;font-style:italic"> <span style="color:red;font-weight:bold">TODO</span> <br>
- rename repo to entity-relation-annotator ?<br>
- find suitable emoji/symbol to put into title <br>
- add license <br>
</ul>

This repository provides a tool for automatically labeling the relationship between two entities in a given sentence using models from the OpenAI API. It takes a CSV file as input, containing sentences along with pre-annotated entity pairs, and returns an updated CSV file with predicted relationship labels. Progress is logged in the terminal throughout the process.

If ground truth labels are provided, the tool can also compute evaluation metrics such as accuracy and Krippendorff’s alpha to assess intercoder reliability. It generates a confusion matrix to visualize model performance and saves all evaluation results to file, enabling easy comparison between different combinations of prompts, models, and datasets.

## Features

- **Batch Processing:** Efficiently processes sentences in batches.
- **Relationship Categorization:** Detect and categorizes relationships using predefined categories:
  - Positive
  - Neutral
  - Negative
  - None  (no relationship detected)
- **CSV Output:** Exports the labeled data in CSV format.
- **Logging**: Logs progress in the terminal.
- **Evaluation Metrics:** Generates confusion matrices and calculates key performance metrics (accuracy, Krippendorff’s Alpha, and Brennan-Prediger’s Alpha) to assess model performance.

## Project Structure

```plaintext
├── data
│   ├── api_output/               # OpenAI-labeled output files
│   ├── evaluation/
│   │   ├── confusion_matrices/   # Confusion matrix plots for analysis
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

  Needed packages are: `tidyr`, `ggplot2`, `gridExtra` and `grid`.

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

2. **Configure the Script**

   In `scripts/run.py`, adjust the following parameters when calling the `process_and_evaluate_files()` function:

   - **model_id**: The OpenAI model identifier (e.g., `gpt-4-turbo`).
   - **system_prompt_file**: The prompt file (located in the `prompts/` folder).
   - **true_value_known**: Boolean indicating if the true relationship value is known and therefore performance metrics can be calculated.
	<p style="background-color:#FFA0F0;font-style:italic"> <span style="color:red;font-weight:bold">TODO</span> add this input parameter to script or add automated checking for such columns. </p>
   - **input_file**: The CSV file name with the input sentences.
   - **batch_size**: Modify the batch size in the `generate_prompts()` function based on the token limits of your chosen model.
   - **num_iterations**: Number of times the sentences should be labelled. If number >1 the metrics can be averaged to make results more robust.
   - **override**: Determines whether to overwrite existing output files or skip them. 

3. **Run the Labeling Process**

```bash
 cd scripts
 python run.py # or python3 run.py on linux/unix
```

<p style="background-color:#FFA0F0;font-style:italic"> <span style="color:red;font-weight:bold">TODO</span> check and adjust detailed output if file still exists</p>
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
- `relation_predicted_head_tail`: Predicted Relationship from entity head to tail.
- `relation_predicted_tail_head`: Predicted Relationship from entity tail to head.
- `relation_true_head_tail`: True relationship from head to tail (if available).
- `relation_true_tail_head`: True relationship from tail to head (if available).


<p style="background-color:#FFA0F0;font-style:italic"> <span style="color:red;font-weight:bold">TODO</span> check and adjust detailed output if file still exists</p>

The detailed output file additionally also contains the following columns:

- `correct_detailed`: Indicates if the detailed relationship labels match.
- `relation_true_simplified`: Simplified version of the true relationship without one-sided/mutual (e.g. `neutral` instead of `neutral1`).
- `relation_predicted_simplified`: Simplified version of the predicted relationship without one-sided/mutual (e.g. `neutral` instead of `neutral1`).
- `correct_simplified`: Indicates if the simplified relationship labels match

## Evaluation Metrics

The evaluation file contains:

- **Dataset Details**:
  - Sample file name and size.
  - Model and prompt used.
- **Performance Metrics**:
  - Accuracy
  - Krippendorff’s Alpha
  - Brennan-Prediger’s Alpha

These metrics help compare different models and prompt configurations.

## Known Issues and Problems with the API
- **Non-deterministic Output:**
Even with `temperature=0` and fixed parameters, OpenAI models can produce slightly different outputs for the same input. To mitigate this, you may run multiple iterations and either average the results or apply a majority voting strategy (see the WIP branch `majority-voting`).

- **Incomplete Responses from API:**
Occasionally, the API may return fewer outputs than expected without raising an error. Re-running the script with the same input usually resolves this. If the issue persists, it may be related to the model’s context window limit. Reducing the batch size or switching to a model with a larger context window typically fixes the problem.

- **Batch Size Sensitivity:**
Large batch sizes may cause truncated outputs or reduced performance, especially on models with smaller context windows (e.g., `gpt-3.5-turbo`). Experiment with batch sizes that fit within the model's token limit.

- **Performance Degradation on Large Jobs:**
Labeling too many files or long inputs in one run can lead to degraded performance, incomplete results, or API rate limiting. For instance, processing 10 files with 50 sentences each has been tested successfully. If you encounter issues, try processing fewer files at a time.

<p style="background-color:#FFA0F0;font-style:italic"> <span style="color:red;font-weight:bold">TODO</span> name example where tool struggled to reliably label sentences</p>