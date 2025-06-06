import json
from helper_functions import *

def main(): 
    # Specify what input file, model and system prompt to use
    model_id = "o1-mini" # Models: gpt-4-turbo, gpt-3.5-turbo, gpt-4o, gpt-4o-2024-11-20, gpt-4o-mini, o1
    system_prompt_file = "bidirectional_prompt.txt"
    input_file = "random_sample_small_5_bidirectional.csv"

    # Specify if the relationship should be labelled uni- or bidirectional
    # Bidirectional labelling only works with specific prompts and samples
    bidirectional_relationship = True # TODO: detect automatically based on prompt name

    data = load_csv(input_file)
    
    batch_size = 50
    
    # Set parameters based on directional type
    if bidirectional_relationship: 
         # Name the output file based on the used parameters for simple identification
        output_file = f"output_{generate_filename(model_id,system_prompt_file,input_file)}_bi.csv"
        prompts = generate_prompts(user_prompt_bidirectional, data, batch_size) # Choose batch size
    else:
        output_file = f"output_{generate_filename(model_id,system_prompt_file,input_file)}_uni.csv"
        prompts = generate_prompts(user_prompt_unidirectional, data, batch_size) # Choose batch size
   
    # Call API with sentences
    results = []

    NUM_SAMPLES = 1  # Number of completions to sample for majority voting
    for i in range(NUM_SAMPLES):
        print(f"Start iteration: {i+1} of {NUM_SAMPLES}")
        for index, prompt in enumerate(prompts):
            print(f"Send batch {index+1} of {len(prompts)} to the OpenAI API...")
            response = generate_relation_labels(prompt, 
                                            system_prompt = system_prompt_file, 
                                            model= model_id, 
                                            temperature = 0)
            if response is None:
                print(f"❌ API returned `None` for batch {index+1}")
            elif response.strip() == "":
                print(f"⚠️ Empty response for batch {index+1}")
            
            if response:
                try:
                    labeled_data = json.loads(response)
                    results.extend(labeled_data)
                    #print(results)
                except json.JSONDecodeError:
                    print("Failed to parse API response as JSON.")
                    print(f"API Response: {response}")
                    # Add empty json if API response fails to not mess up the order of the sentences for the comparison step
                    if bidirectional_relationship:
                        results.extend([{"sentence": "","head": "","tail": "","rel_head_tail": "","rel_tail_head": ""}])
                    else:
                        results.extend([{"sentence": "","head": "","tail": "","relation": ""}])

    save_results_to_csv(results, input_file, output_file, bidirectional=bidirectional_relationship)
    # Aggregate predicted labels and choose most common one
    aggregate_majority_labels(output_file)
    aggregated_file = f"deduplicated_{output_file}"
    # Compare results with true values and generate confusion matrix
    evaluate_model_predictions(model_id, system_prompt_file, input_file, aggregated_file, bidirectional=bidirectional_relationship)
    #evaluate_model_predictions(model_id, system_prompt_file, input_file, output_file, bidirectional=bidirectional_relationship)
    #generate_confusion_matrices(model_id, system_prompt_file, input_file, output_file, bidirectional=bidirectional_relationship, show_plot=False)

if __name__ == "__main__":
    main()
