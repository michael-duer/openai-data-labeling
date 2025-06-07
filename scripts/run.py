import json
from helper_functions import *

def main(): 
    # Specify what input file, model and system prompt to use
    model_id = "o1-mini" # Models: gpt-4-turbo, gpt-3.5-turbo, gpt-4o, gpt-4o-2024-11-20, gpt-4o-mini, o1
    system_prompt_file = "bidirectional_prompt.txt"
    input_file = "random_sample_small_5_bidirectional.csv"

    data = load_csv(input_file)
    
    batch_size = 50
    output_file = f"output_{generate_filename(model_id,system_prompt_file,input_file)}.csv"

    # Create list of prompts based on input file
    prompts = generate_prompts(data, batch_size)
   
    # Send batched prompts to the API and collect predicted relation labels
    results = []
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
                results.extend([{"sentence": "","head": "","tail": "","rel_head_tail": "","rel_tail_head": ""}])

    save_results_to_csv(results, input_file, output_file)
    # Compare results with true values and generate confusion matrix
    evaluate_model_predictions(model_id, system_prompt_file, input_file, output_file)
    generate_confusion_matrix(model_id, system_prompt_file, input_file, output_file, show_plot=False)

if __name__ == "__main__":
    main()
