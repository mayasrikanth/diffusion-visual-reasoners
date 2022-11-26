import json
import inflect
inflect = inflect.engine()
import argparse 


def get_dataset_sizes():
    train_full = './full-gqa-transformed/train_balanced_image_prompt_final.json'
    train_filtered = 'gqa-train-balanced-filtered-new.json'
    val_full = './full-gqa-transformed/val_balanced_image_prompt_final.json'
    val_filtered = 'gqa-val-balanced-filtered-new.json'
    with open(train_full) as f:
        train_full_data = json.load(f)
        print("Number of unique images in train_full: ", len(train_full_data))
    with open(train_filtered) as f:
        train_filtered_data = json.load(f)
        print("Number of unique images in train_filtered: ", len(train_filtered_data))
    with open(val_full) as f:
        val_full_data = json.load(f)
        print("Number of unique images in val_full: ", len(val_full_data))
    with open(val_filtered) as f:
        val_filtered_data = json.load(f)
        print("Number of unique images in val_filtered: ", len(val_filtered_data))


def get_transformed_gqa(sceneGraphsFile, questionsFile, outputFile):
    with open(sceneGraphsFile) as f:
        scene_graphs = json.load(f)
    with open(questionsFile) as f:
        scene_qs = json.load(f)

    image_prompts = build_prompts(scene_qs, scene_graphs)
    with open(outputFile, 'w') as wf:
        wf.write(json.dumps(image_prompts))

# Went to paper to find relevant types; https://arxiv.org/pdf/1902.09506.pdf
def build_prompts(questions, scene_graphs):
    '''
       Iterate through questions in question dictionary. 
       For each question, if it meets certain criteria, pull both the 
       full scene description and the description associated with the 
       objects relevant in the question or answer. 

       Currently filtering for certain types. Comment out the line 
       'if questions[key]['types']['detailed'] in detailed_filter' to 
       keep all types.
       
    '''
    detailed_filter = {'verifyAttr', 'verifyRel', 'chooseAttr', 'exist', 'existRel', \
                      'logicOr', 'logicAnd', 'queryObject', 'queryRel', \
                      'chooseRel', 'chooseObjRel', 'twoDiff'}
        
    image_prompt = {} # mapping between unique image_id: [list of prompts]
    num_captions, num_multi_object = 0, 0
    for key in questions.keys(): # iterate through questions 
        if 'types' in questions[key] and 'detailed' in questions[key]['types']:
            if questions[key]['types']['detailed'] in detailed_filter:
                image_id = questions[key]['imageId']
                if image_id in scene_graphs:
                    if len(questions[key]['annotations']['question']) == 2: # 2 objects
                        obj1, obj2 = questions[key]['annotations']['question'].values() 
                        prompt_str, obj_description, final_image_description = get_object_descriptions(scene_graphs[image_id], obj1, obj2)

                        if len(prompt_str) > 0:
                            # print("SCENE: ", image_id)
                            # print("Question: ", questions[key]['question'])
                            # print("NEW CAPTION, 2 objects")
                            num_captions += 1

                            final_prompt = final_image_description + '\n ' + prompt_str
                            if image_id not in image_prompt:
                                image_prompt[image_id] = [{'prompt': final_prompt, 
                                                           'question_id':key,
                                                           'question': questions[key]['question'], 
                                                           'fullAnswer': questions[key]['fullAnswer'], 
                                                           'type': questions[key]['types']['detailed'], 
                                                           '>2Objects': False}]

                            else:
                                image_prompt[image_id].append({'prompt': final_prompt, 
                                                               'question_id':key,
                                                               'question': questions[key]['question'], 
                                                               'fullAnswer': questions[key]['fullAnswer'], 
                                                               'type': questions[key]['types']['detailed'], 
                                                               '>2Objects': False})

                    elif len(questions[key]['annotations']['question']) == 1: # also allow for 1 object
                        obj1 = list(questions[key]['annotations']['question'].values())[0]
                        obj2 = -1 # some fake id
                        prompt_str, obj_description, final_image_description = get_object_descriptions(scene_graphs[image_id], obj1, obj2)
                        if obj1 in obj_description: # sometimes this isn't the case ...
                            prompt_str = 'Given that ' + obj_description[obj1][:-1].lower() + ', summarize succinctly:'

                            if len(prompt_str) > 0:
                                num_captions += 1

                                final_prompt = final_image_description + '\n ' + prompt_str
                                if image_id not in image_prompt:
                                    image_prompt[image_id] = [{'prompt': final_prompt,
                                                               'question_id':key,
                                                               'question': questions[key]['question'], 
                                                               'fullAnswer': questions[key]['fullAnswer'], 
                                                               'type': questions[key]['types']['detailed'], 
                                                               '>2Objects': False}]
                                else:
                                    image_prompt[image_id].append({'prompt': final_prompt, 
                                                                   'question_id': key,
                                                                   'question': questions[key]['question'], 
                                                                   'fullAnswer': questions[key]['fullAnswer'], 
                                                                   'type': questions[key]['types']['detailed'], 
                                                                   '>2Objects': False})

                    else: # Allow for > 2 objects?
                        num_multi_object += 1
                        objects = list(questions[key]['annotations']['question'].values())
                        prompt_str, obj_description, final_image_description = get_object_descriptions(scene_graphs[image_id], obj1, obj2)

                        relevant_objects = []
                        for obj in objects:
                            if obj in obj_description: 
                                relevant_objects.append(obj_description[obj].lower())

                        if len(relevant_objects) > 0:
                            prompt_str = 'Given that ' + ', '.join(relevant_objects[:-1])
                            prompt_str = prompt_str + ' and ' + relevant_objects[-1][:-1] + ', summarize succinctly:'

                        if len(prompt_str) > 0:
                            num_captions += 1

                            final_prompt = final_image_description + '\n ' + prompt_str
                            if image_id not in image_prompt:
                                image_prompt[image_id] = [{'prompt': final_prompt, 
                                                           'question_id':key, 
                                                           'question': questions[key]['question'], 
                                                           'fullAnswer':questions[key]['fullAnswer'],  
                                                           'type':questions[key]['types']['detailed'], 
                                                           '>2Objects': False}]
                            else:
                                image_prompt[image_id].append({'prompt': final_prompt, 
                                                               'question_id':key,
                                                               'question': questions[key]['question'], 
                                                               'fullAnswer':questions[key]['fullAnswer'], 
                                                               'type':questions[key]['types']['detailed'], 
                                                               '>2Objects': False})
                            

                        # Track number of captions thus far
                        print("Number of captions: ", num_captions)
                        print("Number of images: ", len(image_prompt))

                    
    print(f"{num_multi_object} questions with more than 2 objects.")
    return image_prompt
                            
def get_specific_relation_string(object_relations, object_names, target_obj_id): # why is it repeating itself
    '''
        object_relations is a dictionary of the form {relation_name : target object name},
       for all relations for a gieven obj1.
       
       object_names is a mapping {obj_id : object name (natural language)}
       target_obj_id is the target id for which we want to return the relation from the 
       original object.
       
    '''
    
    temp = []
    weird_relations = {'of', 'with', 'inside'}
    for relation_name in object_relations:
        if target_obj_id in object_names and object_names[target_obj_id] in object_relations[relation_name]: # this returns
            if relation_name in weird_relations:
                continue 
            target_str = ''
            target_str = target_str + relation_name + ' the '

            if len(object_relations[relation_name]) > 1:
                target_str = target_str + ', '.join(object_relations[relation_name][:-1])
                target_str = target_str + ' and ' + object_relations[relation_name][-1]
                target_str = target_str + ''

            else: 
                target_str = target_str + object_relations[relation_name][0]

            temp.append(target_str)

            relation_str = ''
            if len(temp) > 1:
                relation_str = ', '.join(temp[:-1])
                relation_str = relation_str + ', and ' + temp[-1] + '.'
            elif len(temp) == 1: 
                relation_str = temp[0] + '.'
            else:
                relation_str = ''

            return relation_str
    return ''

def get_relation_strings(object_relations, object_name): # why is it repeating itself
    temp = []
    weird_relations = {'of', 'with', 'inside'}
    for relation_name in object_relations:
        if relation_name in weird_relations:
            continue 
        target_str = ''
        target_str = target_str + relation_name + ' the '
         
        if len(object_relations[relation_name]) > 1:
            target_str = target_str + ', '.join(object_relations[relation_name][:-1])
            target_str = target_str + ' and ' + object_relations[relation_name][-1]
            target_str = target_str + ''
        
        else: 
            target_str = target_str + object_relations[relation_name][0]
            
        temp.append(target_str)
       
    relation_str = ''
    if len(temp) > 1:
        relation_str = ', '.join(temp[:-1])
        relation_str = relation_str + ', and ' + temp[-1] + '.'
    elif len(temp) == 1: 
        relation_str = temp[0] + '.'
    else:
        relation_str = ''
          
    return relation_str

def get_relations_dict(relations, object_name):
    '''Return dictionary of form [relation_name : obj_id]'''
    relation_obj = {}
    seen_targets = {} # don't want two relations to the same target
    if len(relations) == 0:
        return {}
    for obj_relation in relations:
        name = obj_relation['name']
        target = object_name[obj_relation['object']] # target object name
        if target in seen_targets:
            continue 
        
        seen_targets[target] = 1
        
        if name in relation_obj:
            relation_obj[name].append(target)
        else:
            relation_obj[name] = [target]
    return relation_obj


def get_object_descriptions(objects, obj1, obj2):
    ''' 
        Returns a relation string describing relationship 
        between obj1 and obj2, as well as a textual description of 
        the entire scene.
        
        'objects' refers to all objects in a given scene.
        'obj1': relevant object id in a question 
        'obj2': the other relevant object id in the question 
        
        Either 'obj1' or 'obj2' could be the source or target. 
        
    '''
    object_name = {}
    object_attributes = {}
    object_relations = {}
    objects = objects['objects']
    for obj_id in objects:
        object_name[obj_id] = objects[obj_id]['name']
        if len(objects[obj_id]['attributes']) == 1:
            object_attributes[obj_id] = objects[obj_id]['attributes'][0]
        elif len(objects[obj_id]['attributes']) == 2:
            object_attributes[obj_id] = ', '.join(objects[obj_id]['attributes'])
        elif len(objects[obj_id]['attributes']) > 2:
            temp = ', '.join(objects[obj_id]['attributes'][:-1])
            temp = temp + ', and ' + objects[obj_id]['attributes'][-1]
            object_attributes[obj_id] = temp 
            
    image_description = []
    prompt_relation_str = ''
    obj_description = {} # mapping from obj_id to single description sentence about object, used for prompting
    for obj_id in object_name:
        temp = 'The ' 
        if obj_id in object_attributes:
            temp += object_attributes[obj_id] 
        temp = temp + ' ' +  object_name[obj_id]
        if inflect.singular_noun(object_name[obj_id]) == False: # check for plural
            temp += ' is '
        else: 
            temp += ' are ' 
        object_relations[obj_id] = get_relations_dict(objects[obj_id]['relations'], object_name)
        # Encode all relations for a single object into a string
        relation_str = get_relation_strings(object_relations[obj_id], object_name)
        
        # Want to get relation string for the two specified objects separately from the entire scene dscription
        if len(prompt_relation_str) == 0:
   
            if obj1 == obj_id:
                prompt_relation_str = get_specific_relation_string(object_relations[obj_id], object_name, obj2)
                # leaving out period in prompt_relation_str
                prompt_relation_str = 'Given that ' + temp.lower() + prompt_relation_str[:-1] + ', summarize succinctly:'
                
            elif obj2 == obj_id:
                prompt_relation_str = get_specific_relation_string(object_relations[obj_id], object_name, obj1)
                # leaving out period in prompt_relation_str
                prompt_relation_str = 'Given that ' + temp.lower() + prompt_relation_str[:-1] + ', summarize succinctly:'

         
        if len(relation_str) == 0:
            continue 
        
        temp = temp + relation_str
        
        obj_description[obj_id] = temp # description of obj_id
        image_description.append(temp)

    final_image_description = ' '.join(image_description)
    return (prompt_relation_str, obj_description, final_image_description)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sceneGraphsFile', type=str, required=False, default='sceneGraphs/train_sceneGraphs.json', 
                        help='.json file containing GQA scene graphs.')

    parser.add_argument('--questionsFile', type=str, required=False, default='questions1.2/train_balanced_questions.json', 
                        help='.json file containing GQA questions.')
    
    parser.add_argument('--outputFile', type=str, required=False, default='GQA_transformed_train.json', 
                        help='name of output .json file with transformed gqa data.')

    args = parser.parse_args()
    
    get_transformed_gqa(args.sceneGraphsFile, args.questionsFile, args.outputFile)
    #get_dataset_sizes()