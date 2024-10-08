import requests
import json


# Configuration for llama 3 API 
class LocalModelConfig:
    def __init__(self):
        self.base_url = "http://127.0.0.1:1234/v1"  #  local server address
        self.api_type = "local_model"
        self.api_key = "not-needed" 

def read_json_file(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# initiate conversation with the local model and handle response.
def initiate_conversation(input_text, system_message, search_results, config):
    headers = {"Content-Type": "application/json"}
    
    # Generate query from search results
    search_content = "\n".join([f"SEARCH RESULT {i+1}: {result['_source']['contents']}" 
                                for i, result in enumerate(search_results['hits']['hits'])])
    
    # making the payload 
    payload = {
    "model": "hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf",
    "messages": [
        {"role": "system", "content": system_message},
        {"role": "user", "content": input_text}
    ],
    "temperature": 0.7,
    "max_tokens": 2048, # can change this value depending on how loong we want answer to be 
    "stream": False
    }

    
    try:
        print("Sending request to the model...")  
        # Send POST request with response handling (streaming disabled)
        response = requests.post(f"{config.base_url}/chat/completions", json=payload, headers=headers)
        print(f"Response status code: {response.status_code}")  
   
        if response.status_code == 200:
            try:
                json_response = response.json()
                full_response = json_response['choices'][0]['message']['content']
                print(f"Full Model Response: {full_response}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing response: {e}")
        else:
            print(f"Non-200 response code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the model: {e}")

def main():

    config = LocalModelConfig()

    # Read system message from file
    system_message_data = read_json_file("system_message.json")
    if system_message_data is None:
        print("System message file not found or invalid.")
        return

    system_message = system_message_data["messages"][0]["content"]

    search_results = read_json_file("search_result.json")
    if search_results is None: # error handling 
        print("Search results file not found or invalid.")
        return

    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'bye', 'end']:
            print("Exiting the conversation.")
            break
        # Pass user input and search results to llama
        initiate_conversation(user_input, system_message, search_results, config)

if __name__ == "__main__":
    main()
