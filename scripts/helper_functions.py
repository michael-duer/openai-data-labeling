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
    """Generate filename based on the input parameters used"""
    # Remove file type from name
    clean_prompt = os.path.splitext(system_prompt_file)[0]
    clean_input = os.path.splitext(input_file)[0]

    return f"{model_id}_{clean_prompt}_{clean_input}"


def load_csv(filename):
    """Load sentences to label from a CSV file inside the data/samples/ folder."""
    filepath = os.path.join("..", "data", "samples", filename)
    data = []
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append({
                "sentence": row["sentence"],
                "head": row["head"],
                "tail": row["tail"]
            })
    return data

def load_system_prompt(filename):
    """Load a system prompt from a file inside the 'prompts' folder."""
    filepath = os.path.join("..", "prompts", filename)
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()

def generate_relation_labels(prompts, system_prompt, model, temperature):
  """Send a prompt to the OpenAI API and return the response."""

  system_prompt = load_system_prompt(system_prompt)

  try:
    # Adjust API request based on model used as reasoning models have a different structure
    if model=="o1-mini": # Currently o1-mini is the only reason model available
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"{system_prompt}\n\n{prompts}"}],
            #temperature=temperature,
            #max_tokens=5000
        )
        text_response = response.choices[0].message.content.strip()
        # If the response is wrapped in markdown formatting (```json), clean it.
        if text_response.startswith("```json"):
            text_response = text_response[7:-3].strip()
        return text_response
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompts}
            ],
            temperature=temperature, # Lower temperature for better accuracy
            )
        # Clean response if it is not well formatted (starts with "```json" and ends with "```")
        if response.choices[0].message.content.startswith("```json"):
            # Remove first 7 and last 3 chars
            cleaned_response = response.choices[0].message.content[7:-3]
            return cleaned_response
        return response.choices[0].message.content
  except Exception as e:
      print(f"Error: {e}")
      return None

def generate_prompts(data, batch_size=5):
    """Generate prompts by bundling sentences into batches."""
    prompts = []
    base_prompt = """
        Please pre-label the following data. Each input consists of:
        Sentence: <text>
        Head: <head entity>
        Tail: <tail entity>

        Determine the relation between the head and tail in each sentence and output the results as a JSON array of objects, where each object has the fields: "sentence", "head", "tail", "relation".
        """
    # base_prompt = (
    #         "Please label the relationships for the following entries according to the instructions provided above. "
    #         "For each entry, determine the relationship (choose one from Positive1, Positive2, Neutral1, Neutral2, Negative1, Negative2, none) "
    #         "between the given 'head' and 'tail' based on the 'sentence'. "
    #         "Return your answer as a JSON array where each element is an object with keys 'sentence', 'head', 'tail', and 'relation'. "
    #         "Do not include any additional text.\n"
    #     )
    # base_prompt = (
    #     "Please label the relationships in the following sentences. The relationships are one of Positive1, Positive2, Neutral1, Neutral2, Negative1, Negative2 or None. The number indicates if the relation ship is one-sided (1) or mutual (2). "
    #     "Provide the result in JSON format with fields 'sentence', 'head', 'tail', and 'relation'.\n"
    #     )
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        prompt = base_prompt
        for item in batch:
            prompt += f"\nSentence: {item['sentence']}\nHead: {item['head']}\nTail: {item['tail']}\n"
        prompts.append(prompt)
    return prompts

def save_results_to_csv(results, input_file, output_file):
    """
    Save labeled results to a CSV file, including true relations from the input file.

    Parameters:
        results (list): List of dictionaries with predicted results.
        input_file (str): Path to the input CSV file containing true relations.
    """
    # Load true relations from the input file
    input_filepath = os.path.join("..", "data", "samples", input_file)
    input_data = pd.read_csv(input_filepath)

    # Convert results (list of dicts) to DataFrame
    results_df = pd.DataFrame(results)

    # Merge results with true relations based on sentence, head, and tail
    merged_df = pd.merge(
        results_df, 
        input_data[["sentence", "head", "tail", "relation"]], 
        on=["sentence", "head", "tail"],
        how="left"
    )
    # Rename columns for clarity
    merged_df.rename(columns={"relation_x": "relation_predicted","relation_y": "relation_true"}, inplace=True)

    # Sometimes the api removes punctuation from the sentences resulting in missing values after the merge.
    # To solve this issue, we identify the NA values and perform a fuzzy match to find the correct sentence, and add the missing true labels 
    missing_true = merged_df["relation_true"].isna()
    # Helper function to find best match only when needed
    def fuzzy_match_missing_rows(row, reference_df):
        """Find best sentence match and fill missing relation."""
        match, score = process.extractOne(row["sentence"], reference_df["sentence"].tolist(), score_cutoff=90)
        if match:
            return reference_df.loc[reference_df["sentence"] == match, "relation"].values[0]
        return None

    # Fill missing values using fuzzy matching
    merged_df.loc[missing_true, "relation_true"] = merged_df[missing_true].apply(
        lambda row: fuzzy_match_missing_rows(row, input_data), axis=1
    )

    # Save the merged results to the output CSV
    output_filepath = os.path.join("..", "data", "api_output", output_file)
    merged_df.to_csv(output_filepath, index=False)
    print(f"Results saved to {output_file}")

def krippendorff_alpha(data, level="detailed"):
    """
    Computes Krippendorff's alpha for inter-rater agreement.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing true and predicted relations.
        level (str): "detailed" for fine-grained labels, "simplified" for mapped categories.

    Returns:
        float: Krippendorff's alpha value.
    """
    if level == "detailed":
        values = data[["relation_true", "relation_predicted"]]
    elif level == "simplified":
        values = data[["relation_true_simplified", "relation_predicted_simplified"]]
    else:
        raise ValueError("Invalid level. Choose 'detailed' or 'simplified'.")

    # Convert categorical labels to numeric encoding
    unique_labels = pd.unique(values.values.ravel()) # Extract unique label categories
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Replace labels with numeric values
    pd.set_option('future.no_silent_downcasting',True)
    values = values.replace(label_mapping).infer_objects(copy=False)

    # Convert to numpy array for Krippendorff calculation
    values = values.to_numpy().T # Transpose to align with Krippendorff's input format

    # Compute Krippendorff's Alpha
    alpha = krippendorff.alpha(reliability_data=values, level_of_measurement='nominal')
    return round(alpha, 4)

def brennan_prediger_alpha(data, level="detailed"):
    """
    Computes Brennan-Prediger's alpha for inter-rater agreement.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing true and predicted relations.
        level (str): "detailed" for fine-grained labels, "simplified" for mapped categories.

    Returns:
        float: Brennan-Prediger's alpha value.
    """
    if level == "detailed":
        true_col, pred_col = "relation_true", "relation_predicted"
        num_classes = 7 # positive1, positive2, neutral1, neutral2, negative1, negative2, none
    elif level == "simplified":
        true_col, pred_col = "relation_true_simplified", "relation_predicted_simplified"
        num_classes = 4 # positive, neutral, negative, none
    else:
        raise ValueError("Invalid level. Choose 'detailed' or 'simplified'.")

    # Calculate observed agreement (accuracy)
    p0 = accuracy_score(data[true_col], data[pred_col]) # TODO FIX ERROR CAUSED BY NaN VALUES

    # Expected agreement assuming equal probability per class
    pe = 1 / num_classes

    # Compute Brennan-Prediger's Alpha
    alpha_bp = (p0 - pe) / (1 - pe)

    return round(alpha_bp, 4)

def evaluate_model_predictions(model_id, system_prompt_file, input_file, output_file):
    """
    Evaluates the model's predictions against the true values in the output file.
    """
    # Load data from the output CSV file
    output_filepath = os.path.join("..", "data", "api_output", output_file)
    data = pd.read_csv(output_filepath)
    
    # Ensure necessary columns are present
    if "relation_true" not in data.columns or "relation_predicted" not in data.columns:
        raise ValueError("The output file must contain 'relation_true' and 'relation_predicted' columns.")
    
    # Lowercase the relations for consistency
    data["relation_true"] = data["relation_true"].str.lower()
    data["relation_predicted"] = data["relation_predicted"].str.lower()
    
    # Evaluate detailed labels
    data["correct_detailed"] = data["relation_true"] == data["relation_predicted"]

    # Mapping for Simplified Labels
    simplify_mapping = {
        "positive1": "positive", "positive2": "positive",
        "neutral1": "neutral", "neutral2": "neutral",
        "negative1": "negative", "negative2": "negative",
        "none": "none"
    }

    # Apply the mapping
    data["relation_true_simplified"] = data["relation_true"].map(simplify_mapping)
    data["relation_predicted_simplified"] = data["relation_predicted"].map(simplify_mapping)
    
    # Evaluate Simplified Labels
    data["correct_simplified"] = data["relation_true_simplified"] == data["relation_predicted_simplified"]

    # Calculate accuracy metrics
    total_count = len(data)
    
    correct_detailed_count = data["correct_detailed"].sum()
    correct_simplified_count = data["correct_simplified"].sum()
    
    accuracy_detailed = round((correct_detailed_count / total_count) * 100, 2)
    accuracy_simplified = round((correct_simplified_count / total_count) * 100, 2)

    # Compute Krippendorf's Alpha
    alpha_detailed = krippendorff_alpha(data, level="detailed")
    alpha_simplified = krippendorff_alpha(data, level="simplified")
    # Compute Brennan-Prediger Alpha
    alpha_bp_detailed = brennan_prediger_alpha(data, level="detailed")
    alpha_bp_simplified = brennan_prediger_alpha(data, level="simplified")

    # Print evaluation to console
    # Define table borders and formatting
    top_border = "\033[1;37m┏" + "━" * 56 + "┓\033[0m"
    middle_border = "\033[1;37m┣" + "━" * 56 + "┫\033[0m"
    bottom_border = "\033[1;37m┗" + "━" * 56 + "┛\033[0m"
    # Title
    print("\n\033[1;4mPre-Labeling Evaluation Summary\033[0m\n")  # Bold and Underlined
    print(f"Input file: {input_file}")
    print(f"Number of samples: {total_count}")
    print(f"Model used: {model_id}")
    print(f"Prompt used: {system_prompt_file}\n")
    print(top_border)
    print(f"\033[1m┃ {'Metric':<24} ┃ {'Full Labels':>11} ┃ {'Simple Labels':>13} ┃\033[0m")
    print(middle_border)
    # Data rows
    print(f"┃ {'Correct Predictions':<24} ┃ \033[32m{correct_detailed_count:>11}\033[0m ┃ \033[32m{correct_simplified_count:>13}\033[0m ┃")  # Green
    print(f"┃ {'Incorrect Predictions':<24} ┃ \033[31m{total_count - correct_detailed_count:>11}\033[0m ┃ \033[31m{total_count - correct_simplified_count:>13}\033[0m ┃")  # Red
    print(f"┃ {'Accuracy':<24} ┃ {accuracy_detailed:>10}% ┃ {accuracy_simplified:>12}% ┃")
    print(f"┃ {'Krippendorff’s Alpha':<24} ┃ {alpha_detailed:>11.3f} ┃ {alpha_simplified:>13.3f} ┃")
    print(f"┃ {'Brennan-Prediger’s Alpha':<24} ┃ {alpha_bp_detailed:>11.3f} ┃ {alpha_bp_simplified:>13.3f} ┃")
    print(bottom_border)

    # Add evaluation data as new row to csv file
    evaluation_filepath = os.path.join("..", "data", "evaluation", "evaluation.csv")
    evaluation_file = pd.read_csv(evaluation_filepath)
    new_row = pd.DataFrame([{
        # Metadata
        "dataset": input_file, "sample_size": total_count, "model": model_id, "prompt": system_prompt_file,
        # Detailed labels
        "accuracy_detailed": accuracy_detailed, "krippendorff_detailed": alpha_detailed, "bp_detailed": alpha_bp_detailed,
        # Simplified labels
        "accuracy_simplified": accuracy_simplified, "krippendorff_simplified": alpha_simplified, "bp_simplified": alpha_bp_simplified
        }])
    evaluation_file = pd.concat([evaluation_file, new_row], ignore_index=True)
    evaluation_file.to_csv(evaluation_filepath, index=False)

    # Optionally, save detailed comparison as a separate CSV
    detailed_output_filepath = os.path.join("..", "data", "api_output", f"detailed_{output_file}")
    data.to_csv(detailed_output_filepath, index=False)

def generate_confusion_matrices(model_id, system_prompt_file, input_file, output_file, show_plot = True):
    """
    Generate and save two confusion matrix plots:
    1. One with detailed labels (positive1, positive2, etc.).
    2. One that simplifies labels into broader categories (positive, neutral, negative, none).

    Parameters:
        output_file (str): Path to the CSV file containing the true and predicted relations.
        show_plot (bool): Whether to display the plots.
    """
    # Load the output CSV file
    output_filepath = os.path.join("..", "data", "api_output", output_file)
    data = pd.read_csv(output_filepath)

    # Check for necessary columns
    if "relation_true" not in data.columns or "relation_predicted" not in data.columns:
        raise ValueError("CSV file must contain 'relation_true' and 'relation_predicted' columns.")
    
    # Lowercase the relations for comparison consistency
    data["relation_true"] = data["relation_true"].str.lower()
    data["relation_predicted"] = data["relation_predicted"].str.lower()
    
    # Fill missing values ##and ensure consistent data types
    data["relation_true"] = data["relation_true"].fillna("none")#.astype(str)
    data["relation_predicted"] = data["relation_predicted"].fillna("none")#.astype(str)

    # --- Detailed Confusion Matrix ---
    labels_detailed = ["positive1", "positive2", "neutral1", "neutral2", "negative1", "negative2", "none"]

    cm_detailed = confusion_matrix(data["relation_true"], data["relation_predicted"], labels=labels_detailed)

    plt.figure(figsize=(15, 7))

    ax1 = plt.subplot(1, 2, 1)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_detailed, display_labels=labels_detailed)
    disp1.plot(cmap=plt.cm.Blues, ax=ax1)
    ax1.set_title('Confusion Matrix (Detailed)', fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label",fontsize=12, labelpad=10.0)
    plt.ylabel("True Label",fontsize=12, labelpad=10.0)

    # --- Simplified Confusion Matrix ---
    # Define mapping for simplification
    simplify_mapping = {
        "positive1": "positive", "positive2": "positive",
        "neutral1": "neutral", "neutral2": "neutral",
        "negative1": "negative", "negative2": "negative",
        "none": "none"
    }
    # Apply mapping
    data["relation_true_simplified"] = data["relation_true"].map(simplify_mapping)
    data["relation_predicted_simplified"] = data["relation_predicted"].map(simplify_mapping)

    labels_simplified = ["positive", "neutral", "negative", "none"]
    cm_simplified = confusion_matrix(data["relation_true_simplified"], data["relation_predicted_simplified"], labels=labels_simplified)

    ax2 = plt.subplot(1, 2, 2)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_simplified, display_labels=labels_simplified)
    disp2.plot(cmap=plt.cm.Oranges, ax=ax2)
    ax2.set_title("Confusion Matrix (Simplified)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label",fontsize=12, labelpad=10.0)
    plt.ylabel("True Label",fontsize=12, labelpad=10.0)

    # Add title containing what model, prompt and sample were used
    clean_prompt_name = os.path.splitext(system_prompt_file)[0]
    clean_sample_name = os.path.splitext(input_file)[0]
    plt.suptitle(f"$\\bf{{Model:}}$ {model_id};     $\\bf{{Prompt:}}$ {clean_prompt_name};     $\\bf{{Sample:}}$ {clean_sample_name}", 
                 fontsize=16, 
                 y=0.98
                 )

    # Display plots if input parameter is set
    plt.tight_layout()
    if show_plot:
        plt.show()
    
    # Save image
    plot_name = f"cm_{generate_filename(model_id,system_prompt_file,input_file)}.png"
    plot_filepath = os.path.join("..","data","evaluation","confusion_matrices", plot_name)
    disp1.figure_.savefig(plot_filepath)