import requests
import os 
import openai 
import json
import argparse 
from random import sample 
openai.api_key = os.getenv("OPENAI_API_KEY")
url = 'https://api.openai.com/v1/completions'
word_limit = 500

# Resources for translating request to python 
# https://github.com/MicrosoftDocs/azure-docs/issues/100533
# https://datagy.io/python-requests-post/

def append_to_json(outFile, image_id, new_entry):
  '''Save captions as you go in case of disconnection. '''
  with open(outFile, 'r') as f:
    data = json.load(f)

  data[image_id] = new_entry 

  with open(outFile, 'w') as wf:
    wf.write(json.dumps(data))
  
def get_LM_captions(dataFile, outFile):
  with open(dataFile) as f: # Load data file
    image_prompts = json.load(f)

  # with open(outFile, 'w') as wf: # in case file isn't there
  #   wf.write(json.dumps({}))

  num_prompts = 0 
  GQA_captions = {}
  print("NUMBER OF IMAGE PROMPTS: ", len(image_prompts))
  # Prefer non-attribute types
  preferred_types = {'verifyRel', 'exist', 'existRel',
                      'logicOr', 'logicAnd', 'queryObject', 'queryRel',
                      'chooseRel', 'chooseObjRel'}

  attribute_types = {'verifyAttr', 'chooseAttr'}

  # Loop through image ids
  for image_id in image_prompts:  # For each image id, get the question with the smallest prompt?
    prompts = image_prompts[image_id]
    chosen_prompt = {} 

    for prompt in prompts:  # first pass for preferred (structure-focused) types
      if prompt['type'] in preferred_types:
        print("Selected a preferred type!")
        chosen_prompt = prompt
        break 

    if len(chosen_prompt) == 0:  
      for prompt in prompts: # second pass for attribute types
        chosen_prompt = prompt
        break 

    # print(chosen_prompt)

    if len(chosen_prompt['prompt'].split()) <= word_limit: # keep cost reasonable... 
      print(chosen_prompt['prompt'].rstrip())
      num_prompts += 1
      print("PROMPT # ", num_prompts)
      
      prompt_dict = {
        "model": "text-curie-001", # should be 'text-curie-001'
        "prompt": chosen_prompt['prompt'].rstrip(),
        "max_tokens": 74, # 65 for val, 73 for train 
        "temperature": 0.7
      }
      # Call get_model_completion to get caption
      LM_caption = get_model_completion(prompt_dict)

      # Save to dictionary (update GQA_captions...)
      if len(LM_caption) > 0:
        GQA_captions[image_id] = chosen_prompt # update... 
        GQA_captions[image_id]['curie_caption'] = LM_caption
        append_to_json(outFile, image_id, GQA_captions[image_id])
      # temporary
      if num_prompts == 23466: #3216: # should be the max ...
        break 

  print("number of captions: ", num_prompts)
  print("NUMBER OF IMAGES: ", len(image_prompts.keys()))

# Anothe question-- 'the' vs 'a'? the picture is next to the .. vs. a picture is next to a?
def get_model_completion(prompt_dict={}):
  '''
    prompt_dict should contain:
      - model: 'text-curie-001' for summarization
      - prompt: single string prompt or list of string prompts 
      - max_tokens: token generation limit (~65 to respect CLIP tokenizer)
      - temperature: 0.7 default (0 for deterministict generations, 1 for stochasticity)

    Example prompt dict:
    prompt_dict = {
      "model": "text-davinci-002", # should be 'text-curie-001'
      "prompt": ["Say this is a test", "tell me i'm cute"],
      "max_tokens": 65,
      "temperature": 0.7
    }

    Example of post request result:
    {'id': 'cmpl-6FREUh6wykSxHr3PEht3y0GBUWsYY', 
    'object': 'text_completion', 
    'created': 1669137622, 
    'model': 'text-davinci-002', 
    'choices': [{'text': '.")\n\nThis is a test.', 'index': 0, 'logprobs': None, 'finish_reason': 'stop'}, 
                {'text': "\n\nYou're cute!", 'index': 1, 'logprobs': None, 'finish_reason': 'stop'}], 
    'usage': {'prompt_tokens': 10, 'completion_tokens': 14, 'total_tokens': 24}}

    This function issues a prompt_dict to the openAI model of choice, 
    iterates through the 'choices' list, saving the 'text' output for each prompt.

    Returns caption to calling function. 
  '''

  headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}'}
  
  try: 
    response = requests.post(url, headers=headers, json=prompt_dict) # THIS WORKS
    print("REQUEST OUTPUT: ", response.text)
    print(type(response.json()))

    result_dict = response.json()
    print(result_dict)
    return result_dict['choices'][0]['text']
    # print(response.ok)
    # print(response.status_code)

  except:  # Some error 
    return ''


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get data/output files for caption creation and saving.')
  parser.add_argument('--dataFile', type=str, required=False, default='gqa-train-balanced-filtered-new.json', 
                      help='.json file containing prompt info.')
  parser.add_argument('--outFile', type=str, required=False, default='gqa_train_balanced_captioned.json', 
                      help='.json file containing prompt info.')
  args = parser.parse_args()

  get_LM_captions(args.dataFile, args.outFile)