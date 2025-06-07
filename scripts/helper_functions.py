from openai import OpenAI
import os
from dotenv import load_dotenv
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import krippendorff
from thefuzz import process

# Load API key from environment variable
load_dotenv()  
if "OPENAI_API_KEY" in os.environ:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
else:
    raise ValueError("Please set the OPENAI_API_KEY in the .env file.")

def generate_filename(model_id,system_prompt_file,input_file):
    """
    Generates a standardized filename based on the model, prompt, and input file used.

    Parameters:
        model_id (str): Identifier for the model used.
        system_prompt_file (str): Filename of the system prompt (e.g., "zero_shot_prompt.txt").
        input_file (str): Filename of the input sample file (e.g., "sample_1.csv").

    Returns:
        str: A formatted filename string (without file extension).
    """
    # Remove file type from name
    clean_prompt = os.path.splitext(system_prompt_file)[0]
    clean_input = os.path.splitext(input_file)[0]

    return f"{model_id}_{clean_prompt}_{clean_input}"

def load_csv(filename):
    """
    Loads sentence data from a CSV file located in the "data/samples/" directory.

    The function reads rows from the CSV and extracts the "sentence", "head", and "tail"
    fields from each entry, returning them as a list of dictionaries.

    Parameters:
        filename (str): Name of the CSV file to load (e.g., "sample_1.csv").

    Returns:
        list: A list of dictionaries, each containing keys: "sentence", "head", and "tail".
    """
    filepath = os.path.join("..", "data", "samples", filename)
    data = pd.read_csv(filepath, usecols=["sentence", "head", "tail"])
    return data.to_dict(orient="records")


def load_system_prompt(filename):
    """
    Loads the content of a system prompt file from the "prompts" directory.

    Parameters:
        filename (str): Name of the prompt file (e.g., "zero_shot_prompt.txt").

    Returns:
        str: The full text content of the system prompt file.
    """
    filepath = os.path.join("..", "prompts", filename)
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()

def clean_api_response(api_response):
    """
    Cleans an API response by removing surrounding Markdown code block formatting.

    Specifically, if the response starts with "```json" and ends with "```", the function
    strips those markers to extract the raw content. If the response is not wrapped in such
    a block, it is returned unchanged.

    Parameters:
        api_response (str): The raw response string returned by the API.

    Returns:
        str: The cleaned response string without Markdown code block formatting, 
             or the original string if no formatting was detected.
    """
    if api_response.startswith("```json") and api_response.endswith("```"):
        # Remove first 7 and last 3 chars
        cleaned_response = api_response[7:-3]
        return cleaned_response
    else:
        return api_response

def generate_relation_labels(prompts, system_prompt, model, temperature):
  """
    Sends a prompt (or batch of prompts) to the OpenAI API to generate predicted relation labels.

    The function handles different model types by adjusting the message structure accordingly.
    It also applies post-processing to clean the response of markdown formatting if present.

    Parameters:
        prompts (str): One or more input prompts formatted for the model (typically generated in batches).
        system_prompt (str): Filename or string containing the system prompt text.
        model (str): Identifier of the OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo", "o1-mini").
        temperature (float): Sampling temperature for the generation (lower values = more deterministic output).

    Returns:
        str or None: The cleaned response from the API as a string, or None if an error occurred.
    """
  system_prompt = load_system_prompt(system_prompt)

  try:
    # Adjust API request based on model used as reasoning models use a different structure
    if model=="o1-mini" or "o3-mini":
        response = client.chat.completions.create(
            model = model,
            #reasoning_effort = "low", # Possible options: low, medium, high
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{prompts}"}],
        )
        text_response = response.choices[0].message.content
        cleaned_response = clean_api_response(text_response)
        return cleaned_response
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompts}
            ],
            temperature=temperature, # Lower temperature for better accuracy
            #top_p=0,
            )
        text_response = response.choices[0].message.content
        cleaned_response = clean_api_response(text_response)
        return cleaned_response
  except Exception as e:
      print(f"Error: {e}")
      return None

base_prompt = """
    Please label the provided sentences according to the relation categories defined above.
    For each sentence, clearly identify:
    - the relation from "head" to "tail"
    - and the relation from "tail" to "head".

    Provide your response strictly as a JSON array of objects:
    [
        {
            "sentence": "<original sentence>",
            "head": "<head entity or empty string>",
            "tail": "<tail entity or empty string>",
            "rel_head_tail": "<relation label head → tail>",
            "rel_tail_head": "<relation label tail → head>"
        },
        ...
    ]

    Do not provide any additional commentary or explanations.
    """
def generate_prompts(data, batch_size=5):
    """
    Generates formatted prompts by bundling sentence data into batches of specified size.

    Parameters:
        data (list): A list of dictionaries, where each dictionary contains 'sentence', 'head', and 'tail' keys.
        batch_size (int, optional): Number of examples to include in each prompt. Defaults to 5.

    Returns:
        list: A list of prompt strings, each containing a batch of formatted examples.
    """
    prompts = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        prompt = base_prompt
        for item in batch:
            prompt += f"\nSentence: {item['sentence']}\nHead: {item['head']}\nTail: {item['tail']}\n"
        prompts.append(prompt)
    return prompts

def save_results_to_csv(results, input_file, output_file):
    """
    Save labeled results to a CSV file, including ground truth labels from the input file.

    Parameters:
        results (list): List of dictionaries containing model predictions. Each dictionary 
                        must include 'sentence', 'head', 'tail', 'rel_head_tail', and 'rel_tail_head'.
        input_file (str): Filename of the CSV file containing true relation labels. 
                          Must be located in '../data/samples/'.
        output_file (str): Filename for the resulting CSV file. Will be saved to 
                           '../data/api_output/'.
    """
    # Load true relations from the input file
    input_filepath = os.path.join("..", "data", "samples", input_file)
    input_data = pd.read_csv(input_filepath)

    # Convert results (list of dicts) to DataFrame
    results_df = pd.DataFrame(results)

    # Merge results with true relations based on sentence, head, and tail
    merged_df = pd.merge(
        results_df,
        input_data[["sentence", "head", "tail", "rel_head_tail", "rel_tail_head"]],
        on=["sentence", "head", "tail"],
        how="left"
    )
    merged_df.rename(columns={
        "rel_head_tail_x": "relation_predicted_head_tail",
        "rel_tail_head_x": "relation_predicted_tail_head",
        "rel_head_tail_y": "relation_true_head_tail",
        "rel_tail_head_y": "relation_true_tail_head"
    }, inplace=True)

    # Sometimes the api removes punctuation from the sentences resulting in missing values after the merge.
    # To solve this issue, we identify the NA values and perform a fuzzy match to find the correct sentence, and add the missing true labels 
    missing_true = merged_df["relation_true_head_tail"].isna() | merged_df["relation_true_tail_head"].isna()
    # Helper function to find best match only when needed
    def fuzzy_match(row, reference_df):
        """Find best sentence match and fill missing relation."""
        match, score = process.extractOne(row["sentence"], reference_df["sentence"].tolist(), score_cutoff=90)
        if match:
            matched_row = reference_df.loc[reference_df["sentence"] == match]
            return (
                matched_row["rel_head_tail"].values[0],
                matched_row["rel_tail_head"].values[0]   
            )
        return (None, None)

    # Fill missing values using fuzzy matching
    merged_df.loc[missing_true, ["relation_true_head_tail", "relation_true_tail_head"]] = merged_df[missing_true].apply(
        lambda row: pd.Series(fuzzy_match(row, input_data), 
                                index=["relation_true_head_tail", "relation_true_tail_head"]),
                                axis=1)

    # Save the merged results to the output CSV
    output_filepath = os.path.join("..", "data", "api_output", output_file)
    merged_df.to_csv(output_filepath, index=False)
    print(f"Results saved in {output_filepath}")

def krippendorff_alpha(data, column_true, column_predicted):
    """
    Computes Krippendorff's alpha to measure inter-rater agreement between 
    true and predicted categorical labels.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing true and predicted relations.
    Returns:
        float: Krippendorff's alpha value.
    """
    values = data[[column_true, column_predicted]]

    # Convert categorical labels to numeric encoding
    unique_labels = pd.unique(values.values.ravel()) # Extract unique label categories
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Replace labels with numeric values
    pd.set_option("future.no_silent_downcasting",True) # Silence warning from pandas about downcasting
    values = values.replace(label_mapping).infer_objects(copy=False)

    # Convert to numpy array for Krippendorff calculation
    values = values.to_numpy().T # Transpose to align with Krippendorff's input format

    # Compute Krippendorff's Alpha
    alpha = krippendorff.alpha(reliability_data=values, level_of_measurement="nominal")
    return round(alpha, 4)

def brennan_prediger_alpha(data, column_true, column_predicted):
    """
    Computes Brennan-Prediger's alpha, a measure of inter-rater agreement 
    that adjusts for chance agreement assuming uniform class distribution.
    
    Parameters:
        data (pd.DataFrame): A DataFrame containing the annotations.
        column_true (str): Name of the column with ground truth labels.
        column_predicted (str): Name of the column with predicted labels.

    Returns:
        float: Brennan-Prediger's alpha, rounded to 4 decimal places.
               Value ranges from -1 (complete disagreement) to 1 (perfect agreement),
               with 0 indicating agreement no better than chance under uniform class assumptions.
    """
    NUM_CLASSES = 4 # positive, neutral, negative, none
    
    # Calculate observed agreement (accuracy)
    p0 = accuracy_score(data[column_true], data[column_predicted])

    # Expected agreement assuming equal probability per class
    pe = 1 / NUM_CLASSES

    # Compute Brennan-Prediger's Alpha
    alpha_bp = (p0 - pe) / (1 - pe)

    return round(alpha_bp, 4)

def evaluate_model_predictions(model_id, system_prompt_file, input_file, output_file):
    """
    Evaluates a model's predicted relations against ground truth values.

    This function performs the following:
    - Loads prediction results from a CSV file.
    - Computes accuracy, Krippendorff’s alpha, and Brennan-Prediger’s alpha.
    - Displays a formatted evaluation summary in the terminal.
    - Appends the evaluation results to a central evaluation CSV log.
    - Generates a detailed per-example output CSV indicating correctness 
      for both prediction directions.

    Parameters:
        model_id (str): Identifier for the model used.
        system_prompt_file (str): Filename of the system prompt used in the API call.
        input_file (str): Original input dataset file name (for metadata tracking).
        output_file (str): Output file containing model predictions to evaluate.

    Raises:
        ValueError: If any of the required prediction/label columns are missing.
    """
    # Load data
    output_filepath = os.path.join("..", "data", "api_output", output_file)
    data = pd.read_csv(output_filepath)

    total_count = len(data)

    # Ensure necessary columns are present
    if "relation_true_head_tail" not in data.columns or \
        "relation_true_tail_head" not in data.columns or \
        "relation_predicted_head_tail" not in data.columns or \
        "relation_predicted_tail_head" not in data.columns:
        raise ValueError("The output file must contain the columns: 'relation_true_head_tail', 'relation_true_tail_head', 'relation_predicted_head_tail' and 'relation_predicted_tail_head'.")

    # Collect both directions into a long-form DataFrame
    long_data = pd.DataFrame({
        "relation_true": pd.concat([
            data["relation_true_head_tail"],
            data["relation_true_tail_head"]
        ], ignore_index=True),
        "relation_predicted": pd.concat([
            data["relation_predicted_head_tail"],
            data["relation_predicted_tail_head"]
        ], ignore_index=True)
    })

    # Normalize labels
    long_data["relation_true"] = long_data["relation_true"].astype(str).str.lower()
    long_data["relation_predicted"] = long_data["relation_predicted"].astype(str).str.lower()

    # Compute correctness of predicted labels
    long_data["correct"] = long_data["relation_true"] == long_data["relation_predicted"]

    correct_count = long_data["correct"].sum()
    accuracy = round((correct_count / len(long_data)) * 100, 2)

    k_alpha = krippendorff_alpha(long_data, "relation_true", "relation_predicted")
    bp_alpha = brennan_prediger_alpha(long_data, "relation_true", "relation_predicted")

    # Print summary to console
    print("\n\033[1;4mPre-Labeling Evaluation Summary\033[0m\n")
    print(f"Input file: {input_file}")
    print(f"Number of sentences: {total_count}")
    print(f"Model: {model_id}")
    print(f"Prompt: {system_prompt_file}\n")

    print("\033[1;37m┏" + "━" * 56 + "┓\033[0m")
    print(f"\033[1m┃ {'Metric':<24} ┃ {'Value':>27} ┃\033[0m")
    print("\033[1;37m┣" + "━" * 56 + "┫\033[0m")
    print(f"┃ {'Correct Predictions':<24} ┃ \033[32m{correct_count:>27}\033[0m ┃")
    print(f"┃ {'Incorrect Predictions':<24} ┃ \033[31m{len(long_data) - correct_count:>27}\033[0m ┃")
    print(f"┃ {'Accuracy':<24} ┃ {accuracy:>26}% ┃")
    print(f"┃ {'Krippendorff’s Alpha':<24} ┃ {k_alpha:>27.3f} ┃")
    print(f"┃ {'BP Alpha':<24} ┃ {bp_alpha:>27.3f} ┃")
    print("\033[1;37m┗" + "━" * 56 + "┛\033[0m")

    # Log results to evaluation.csv
    evaluation_filepath = os.path.join("..", "data", "evaluation", "evaluation.csv")
    evaluation_file = pd.read_csv(evaluation_filepath)
    new_row = pd.DataFrame([{
        # Metadata
        "dataset": input_file, "sample_size": total_count, "model": model_id, "prompt": system_prompt_file,
        # Metrics
        "accuracy": accuracy, "krippendorff_alpha": k_alpha, "bp_alpha": bp_alpha
        }])
    evaluation_file = pd.concat([evaluation_file, new_row], ignore_index=True)
    evaluation_file.to_csv(evaluation_filepath, index=False)

    # Save detailed comparison as a separate CSV
    detailed_output_filepath = os.path.join("..", "data", "api_output", f"detailed_{output_file}")
    # Insert column indicating if prediction is true/false
    data["correct_head_tail"] = data["relation_predicted_head_tail"] == data["relation_true_head_tail"]
    data["correct_tail_head"] = data["relation_predicted_tail_head"] == data["relation_true_tail_head"]
    # Adjust order of columns
    data = data[list(("sentence",
                        "head",
                        "tail",
                        "relation_predicted_head_tail",
                        "relation_true_head_tail",
                        "correct_head_tail",
                        "relation_predicted_tail_head",
                        "relation_true_tail_head",
                        "correct_tail_head",
                        ))]
    data.to_csv(detailed_output_filepath, index=False)

def generate_confusion_matrix(model_id, system_prompt_file, input_file, output_file, show_plot = True):
    """
    Generates and saves a confusion matrix plot for model predictions.

    Parameters:
        model_id (str): Identifier of the model used.
        system_prompt_file (str): Filename of the system prompt used to generate predictions.
        input_file (str): Original input dataset filename (used in plot title and filename).
        output_file (str): Filename of the CSV containing the predicted and true labels.
                           Must include the columns 'relation_true' and 'relation_predicted'.
        show_plot (bool, optional): Whether to display the confusion matrix plot interactively. 
                                    Defaults to True.

    Raises:
        ValueError: If 'relation_true' or 'relation_predicted' columns are missing in the CSV.
    """
    # Load the output CSV file
    output_filepath = os.path.join("..", "data", "api_output", output_file)
    data = pd.read_csv(output_filepath)

    # Ensure necessary columns are present
    if "relation_true_head_tail" not in data.columns or \
        "relation_true_tail_head" not in data.columns or \
        "relation_predicted_head_tail" not in data.columns or \
        "relation_predicted_tail_head" not in data.columns:
        raise ValueError("The output file must contain the columns: 'relation_true_head_tail', 'relation_true_tail_head', 'relation_predicted_head_tail' and 'relation_predicted_tail_head'.")

    # ---- Prepare Data ----
    # Collect both directions into a long-form DataFrame
    long_data = pd.DataFrame({
        "relation_true": pd.concat([
            data["relation_true_head_tail"],
            data["relation_true_tail_head"]
        ], ignore_index=True),
        "relation_predicted": pd.concat([
            data["relation_predicted_head_tail"],
            data["relation_predicted_tail_head"]
        ], ignore_index=True)
    })

    # Normalize labels
    long_data["relation_true"] = long_data["relation_true"].astype(str).str.lower()
    long_data["relation_predicted"] = long_data["relation_predicted"].astype(str).str.lower()
    
    # Fill missing values and ensure consistent data types
    long_data["relation_true"] = long_data["relation_true"].fillna("NA").astype(str) #TODO is fill NA a smart solution??
    long_data["relation_predicted"] = long_data["relation_predicted"].fillna("NA").astype(str)

    # ---- Create Plot ----
    # Set plot size
    plt.rcParams["figure.figsize"] = (9,8)
    # Create Confusion Matrix
    labels = ["positive","neutral","negative","none"]
    cm = confusion_matrix(long_data["relation_true"], long_data["relation_predicted"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    
    # Add title and labels
    plt.title("Confusion Matrix", fontsize=15, pad=20)
    plt.xlabel("Predicted Label",fontsize=12, labelpad=10.0)
    plt.ylabel("True Label",fontsize=12, labelpad=10.0)

    # Add information about what model, prompt and sample were used
    clean_prompt_name = os.path.splitext(system_prompt_file)[0]
    clean_sample_name = os.path.splitext(input_file)[0]
    plt.suptitle(f"$\\bf{{Model:}}${model_id};  $\\bf{{Prompt:}}${clean_prompt_name};  $\\bf{{Sample:}}${clean_sample_name}", 
                 fontsize=12, 
                 y=0.03
                 )

    # Display plots if input parameter is set
    plt.tight_layout()
    if show_plot:
        plt.show()
    
    # Save plot as image
    plot_name = f"cm_{generate_filename(model_id,system_prompt_file,input_file)}.png"
    plot_filepath = os.path.join("..","data","evaluation","confusion_matrices", plot_name)
    disp.figure_.savefig(plot_filepath)