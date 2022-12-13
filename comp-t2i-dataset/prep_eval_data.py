# write script to generate drawbench samples glide 
import pandas as pd
import jsonlines 
import os
import pickle 
import json
import re 

REPO_PATH = '/home/mayashar/Desktop/diffusion-visual-reasoners/comp-t2i-dataset/'


def get_drawbench_prompts(filename='/home/mayashar/Desktop/pyglide/ImagenDrawBenchPrompts.csv'): # FIX directory
    df = pd.read_csv(filename)
    prompt_types = ['Counting', 'Positional'] 
    df = df[df["Category"].isin(prompt_types)]
    prompts = df['Prompts']
    print("NUMBER OF DRAWBENCH PROMPTS: ")
    print(len(prompts))
    return prompts

def get_prompts_winoground(json_file='/home/mayashar/Desktop/pyglide/winoground_test_seen.json'):
    with open(json_file) as f:
        data = json.load(f)
    prompts = []
    for img_id in data:
        prompts.append(data[img_id])
    return prompts


def get_ex_data_winoground(): # TEST NOW
    ex_data = {}
    ex_id_pairs = []

    winoground_prompts = get_prompts_winoground()
    for prompt in winoground_prompts:
        temp_prompt = prompt.rstrip()
        temp_prompt = temp_prompt.replace("'", " ")
        temp_prompt = temp_prompt.replace(",", " ")
        temp_prompt = temp_prompt.replace("-", " ")
        

        prompt_words = temp_prompt.split(" ")
        print(prompt_words)
        #prompt_words = temp_prompt.split()
        image_name = '_'.join(prompt_words)
        # image_name = image_name.replace("__", "_")
        #image_name = #image_name[:-1] # get rid of period
        image_name = image_name.lower()
        #image_name += '_'
        ex_data[image_name] = {0: {'text': prompt, 'heldout_pairs': []}}


    ex_split = {'all': list(ex_data.keys())}

    return ex_data, ex_split



def get_ex_data_drawbench(): # TEST NOW
    ex_data = {}
    ex_id_pairs = []

    # Load all captions
    # Insert underscores between spaces + png for the image name 
    drawbench_prompts = get_drawbench_prompts()
    for prompt in drawbench_prompts:
        temp_prompt = prompt.rstrip()
        prompt_words = temp_prompt.split()
        image_name = '_'.join(prompt_words)
        image_name = image_name[:-1] # get rid of period
        image_name = image_name.lower()
        image_name += '_'
        ex_data[image_name] = {0: {'text': prompt, 'heldout_pairs': []}}


    ex_split = {'all': list(ex_data.keys())}

    return ex_data, ex_split


# def get_ex_data_wino():
#     ex_data = {}
#     ex_id_pairs = []
#     winogrand_caption_dir = f'{REPO_PATH}winogrand/examples.jsonl'
#     winogrand_split_dir = f'{REPO_PATH}winogrand/winoground_test_seen.json'
    
#     with open(winogrand_split_dir) as f: # Get test_seen winogrand split for eval
#         winogrand_eval_split = json.load(f)

#     with jsonlines.open(winogrand_caption_dir) as reader:
#         for obj in reader:
#             print(obj)
#             if obj['id'] == 399:
#                 continue
#             if obj['image_0'] in winogrand_eval_split:
#                 ex_data[obj['image_0']] = {0: {'text': obj['caption_0'], 'heldout_pairs': []}}
#             if obj['image_1'] in winogrand_eval_split:
#                 ex_data[obj['image_1']] = {0: {'text': obj['caption_1'], 'heldout_pairs': []}}
#             # construct pair list for train/test split
#             #ex_id_pairs.append((obj['image_0'], obj['image_1']))

    # return ex_data 
# TODO: change GENERATED_DIR to the directory with all image generations for winogrand
# TODO: as images are generated, save them with the original image id names 
# TODO: generate pkl file with the winogrand caption generations 

# TODO: do the same for drawbench captions!

def save_eval_pkl_file(GENERATED_DIR='path_to_folder_containing_generated_images',
                       outfile='eval_data/drawbench_glide_laion_FTwino.pkl',
                       eval_data='winoground'):
    
    # Code snippets from jupyter 
    #GENERATED_DIR = '/home/jasonlin/repos/datasets/t2i_benchmark/winoground/finetuned_v0/'
    if eval_data == 'winoground':
        ex_data, ex_split = get_ex_data_winoground()
    elif eval_data == 'drawbench':
        ex_data, ex_split = get_ex_data_drawbench()
    # format: img_id, cap_id, gen_img_path, r_precision_prediction
    pred_data = [(k, 0, os.path.join(GENERATED_DIR, f'{k}.png'), -1) for k, v in ex_data.items()]
 
    with open(outfile, 'wb') as pred_file:
        pickle.dump(pred_data, pred_file)
    #with open('predictions/winoground_all_img_v1-5_ftv0.pkl', 'wb') as pred_file:
        pickle.dump(pred_data, pred_file)

    # Save data.pkl
    with open(f'{REPO_PATH}eval_data/data.pkl', 'wb') as data_file:
        pickle.dump(ex_data, data_file)
        
    # save split,pkl 
    with open(f'{REPO_PATH}eval_data/split.pkl', 'wb') as split_file:
        pickle.dump(ex_split, split_file)




if __name__ == "__main__":
    #get_drawbench_prompts()
    #save_eval_pkl_file(GENERATED_DIR='/home/mayashar/Desktop/pyglide/glide_outputs_laion/sr/', outfile='eval_data/drawbench_glide_laion_noFT.pkl', eval_data='drawbench')
    #save_eval_pkl_file(GENERATED_DIR='/home/mayashar/Desktop/pyglide/glide_outputs_laion_FTwino/sr/', outfile='eval_data/drawbench_glide_laion_FTwino.pkl', eval_data='drawbench')
    #save_eval_pkl_file(GENERATED_DIR='/home/mayashar/Desktop/pyglide/glide-laion-noFT_wino_outputs/sr/', outfile='eval_data/winoground_glide_laion_noFT.pkl')
    save_eval_pkl_file(GENERATED_DIR='/home/mayashar/Desktop/pyglide/glide-laion-FTwino_wino_outputs/sr/', outfile='eval_data/winoground_glide_laion_FTwino.pkl')
# Categories to filter for: (sample 1 image per prompt)
# output to appropriate directory
# load model once, iterate through prompts
# use default glide for now
# def get_prompts(filename='ImagenDrawBenchPrompts.csv'):
#     df = pd.read_csv(filename)
#     prompt_types = ['Counting', 'Positional'] #, 'Colors'] # 'Colors',
#     df = df[df["Category"].isin(prompt_types)]
#     prompts = df['Prompts']
#     print(prompts)
#     print(len(prompts))