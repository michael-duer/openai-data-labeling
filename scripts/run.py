import json
import os
from tqdm import tqdm
from helper_functions import *

def process_and_evaluate_files(model_id, test_files, system_prompt, batch_size = 10, num_iterations = 1, override = False):
    """
    Run API-based relation labeling on multiple input files using a specified model and prompt.

    Parameters:
    - model_id: The OpenAI model to use.
    - test_files: List of input CSV files to process.
    - system_prompt_file: Path to the system prompt text file.
    - batch_size: Number of rows per batch sent to the API.
    - num_iterations: Number of times to run the loop (for averaging).
    - override: If True, reprocess files even if output already exists.
    """

    # Loop over list of test files
    for file_index, input_file in enumerate(test_files):
        # Generate output name and check if name already exists in output folder
        output_file = f"output_{generate_filename(model_id,system_prompt,input_file)}.csv"
        output_dir = os.path.abspath(os.path.join("..", "data", "api_output"))
        output_filepath = os.path.join(output_dir, output_file)
        file_exists = os.path.exists(output_filepath)

        # If there is no file or if override is set to true -> create new file
        if file_exists and not override:
            print(f"Skipping {input_file} -> Output file already exists.")
            continue

        print("------------------------------------------------------------")
        print(f"[START] Processing: {input_file}\n")
        #print("------------------------------------------------------------\n")
        
        # Load files
        data = load_csv(input_file)
        # Save number to compare with output
        num_sentences_input = len(data)

        # Create list of prompts based on input file
        prompts = generate_prompts(data, batch_size)
    
        # Send batched prompts to the API and collect predicted relation labels
        results = []
        for batch_index, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Processing batches"):
            #print(f"  > Processing batch {batch_index + 1} of {len(prompts)}")
            response = generate_relation_labels(prompt, 
                                            system_prompt = system_prompt, 
                                            model = model_id, 
                                            temperature = 0
                                            )
            # TODO improve error handling
            if response is None:
                print(f"❌ API returned `None` for batch {batch_index+1}")
                # Execute response = generate_relation_labels again?
            elif response.strip() == "":
                print(f"⚠️ Empty response for batch {batch_index+1}")
            
            if response:
                try:
                    labeled_data = json.loads(response)
                    results.extend(labeled_data)
                    #print(results)
                except json.JSONDecodeError:
                    print("Failed to parse API response as JSON.")
                    print(f"API Response: {response}")
                    # Add empty json if API response fails to not mess up the order of the sentences for the comparison step
                    results.extend([{"sentence": "","head": "","tail": "","rel_head_tail": "","rel_tail_head": ""}])

        save_results_to_csv(results, input_file, output_file)

        # Compare number of input and output sentences
        output_data = load_csv(output_filepath)
        num_sentences_output = len(output_data)

        if num_sentences_input != num_sentences_output:
            print(f"⚠️ Mismatch in numer of sentences: input = {num_sentences_input}, output = {num_sentences_output}")
            user_input = input("Do you want to re-label this file? [y/n] ")
            if user_input.strip().lower() == "y":
                print("🔁 Rerun not yet implemented.")
                # TODO implement re-run functionality
                # Could re-call this function with override=True for this file

        # Compare results with true values and generate confusion matrix
        evaluate_model_predictions(model_id, system_prompt, input_file, output_file)
        generate_confusion_matrix(model_id, system_prompt, input_file, output_file, show_plot=False)

        #print("------------------------------------------------------------")
        print(f"[END] Finished processing file {file_index+1} of {len(test_files)}.")
        print("------------------------------------------------------------\n")

    print("All files have been processed.")

def main(): 
    # Specify model and system prompt to use
    model_id = "o4-mini" # Models: gpt-4-turbo, gpt-3.5-turbo, gpt-4o, gpt-4o-2024-11-20, gpt-4o-mini, o1
    system_prompt_file = "bidirectional_prompt.txt" # old_prompt.txt bidirectional_prompt.txt new_prompt2.txt

    # 10 small samples with 50 sentences each
    test_files_small = ["random_sample_small_3.csv","random_sample_small_4.csv","random_sample_small_5.csv",
                        "random_sample_small_6.csv","random_sample_small_7.csv","random_sample_small_8.csv",
                        "random_sample_small_9.csv","random_sample_small_10.csv","random_sample_small_11.csv",
                        "random_sample_small_12.csv"]
    
    # 2 bigger samples with 500 sentences each
    test_files_large = ["random_sample_11.csv", "random_sample_12.csv"]

    # Single files must be inside a list for the function to work properly
    trumpiverse_article = ["forbes_trumpiverse.csv"]

    # Call API with set parameters on chosen set of files
    # Using too many files/sentences might lead to rate limiting and worse performance 
    process_and_evaluate_files(model_id, trumpiverse_article, system_prompt_file, batch_size = 50, num_iterations = 1, override = False)


if __name__ == "__main__":
    main()
