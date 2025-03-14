import json
from helper_functions import *

def main(): 
    # Specify what input file, model and system prompt to use
    model_id = "o1-mini" #"gpt-4-turbo" # Models: gpt-4-turbo, gpt-3.5-turbo, gpt-4o, gpt-4o-2024-11-20, gpt-4o-mini, o1
    system_prompt_file = "reasoning_prompt.txt"
    input_file = "./samples/random_sample_long.csv"
    output_file = "output.csv"

    data = load_csv(input_file)
    prompts = generate_prompts(data, batch_size=5)

    results = []
    for index, prompt in enumerate(prompts):
        print(f"Sending batch {index+1} out of {len(prompts)} to OpenAI API...")
        response = generate_relation_labels(prompt, 
                                         system_prompt = system_prompt_file, 
                                         model= model_id, 
                                         temperature = 0)
        if response:
            try:
                labeled_data = json.loads(response)
                results.extend(labeled_data)
            except json.JSONDecodeError:
                print("Failed to parse API response as JSON.")
                print(f"API Response: {response}")
                # Add empty json if API response fails to not mess up the order of the sentences for the comparison step
                results.extend([{"sentence": "","head": "","tail": "","relation": ""}])

    save_results_to_csv(results, input_file, output_file)
    # Compare results with true values and show confusion matrix
    evaluate_model_predictions(model_id, system_prompt_file, input_file, output_file)
    generate_confusion_matrices(output_file, show_plot=False)

if __name__ == "__main__":
    main()
