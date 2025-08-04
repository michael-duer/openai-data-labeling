from helper_functions import *

def main(): 
    # Specify model and system prompt to use
    model_id = "o4-mini" # gpt-4-turbo, gpt-3.5-turbo, gpt-4o, gpt-4.1, gpt-4o-mini, o1, o4-mini
    system_prompt_file = "detailed_guidance_prompt.txt" # "zero_shot_prompt.txt" "detailed_guidance_prompt.txt" "moderate_guidance_prompt.txt"
    
    # 10 small samples with 50 sentences each
    test_files_small = ["random_sample_small_3.csv","random_sample_small_4.csv","random_sample_small_5.csv",
                        "random_sample_small_6.csv","random_sample_small_7.csv","random_sample_small_8.csv",
                        "random_sample_small_9.csv","random_sample_small_10.csv","random_sample_small_11.csv",
                        "random_sample_small_12.csv"]
    
    # 2 bigger samples with 500 sentences each
    test_files_large = ["random_sample_11.csv", "random_sample_12.csv"]

    # Single files must be inside a list for the function to work properly
    trumpiverse_article = ["forbes_trumpiverse.csv"]
    synthetic_data = ["synthetic_sample.csv"]

    # Call API with set parameters on chosen set of files
    # Using too many files/sentences might lead to rate limiting and worse performance 
    process_and_evaluate_files(model_id, trumpiverse_article, system_prompt_file, batch_size = 50, override = True)


if __name__ == "__main__":
    main()
