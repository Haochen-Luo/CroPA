#%%
import os
import json
import random
import numpy as np
import torch
import GPUtil
from utils.eval_tools import train_question
from utils.eval_datasets import  CroPADataset
from huggingface_hub import login

CONFIG_PATH = "data/config.json"

# %%
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_available_gpus(min_memory_available):
    """Returns a list of IDs for GPUs with more than min_memory_available MB of memory."""
    GPUs = GPUtil.getGPUs()
    available_gpus = [gpu.id for gpu in GPUs if gpu.memoryFree > min_memory_available]
    assert len(available_gpus) > 0, "No GPUs available"
    print(f"Using GPUs: {available_gpus[0]} available {available_gpus}")
    return available_gpus

def add_extra_args(args, model_name, config_path = CONFIG_PATH):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    if model_name == "open_flamingo":
        args.prompt_num_to_alpha2 = {1:0.01,5:0.0025,10:0.01,50:0.01,100:0.01}
        args.eoc = "<|endofchunk|>"
    elif model_name == "blip2":
        args.prompt_num_to_alpha2 = {1:0.01,5:0.025,10:0.01,50:0.015,100:0.015}
        args.eoc="</s>"
    elif model_name == "instructblip":
        args.prompt_num_to_alpha2 = {1:0.01,5:0.025,10:0.01,50:0.015,100:0.015}
        args.eoc = " </s>"        
    else:
        raise NotImplementedError(f"{model_name} not implemented, only support open_flamingo, blip2 and instructblip")
        
    VQA_ROOT = config["data_root"]    
    args.vqav2_train_image_dir_path = f"{VQA_ROOT}/train2014"
    args.vqav2_test_image_dir_path = f"{VQA_ROOT}/val2014"                      
    args.vqav2_train_annotations_json_path = f"{VQA_ROOT}/v2_mscoco_train2014_annotations.json"
    args.vqav2_train_questions_json_path = f"{VQA_ROOT}/v2_OpenEnded_mscoco_train2014_questions.json"
    args.vqav2_eval_annotations_json_path = f"{VQA_ROOT}/v2_mscoco_val2014_annotations.json"
    
    args.test_annotations_json_path = "data/filtered_v2_mscoco_val2014_annotations.json"
    args.test_questions_json_path = "data/filtered_v2_OpenEnded_mscoco_val2014_questions.json"
    args.eval_batch_size = config["eval_batch_size"]

def compute_effective_num_shots(num_shots, model_type):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots

def get_intended_token_ids(input_ids, target_id,debug = False):
    padding = torch.full_like(input_ids, -100)
    padding_dim = padding.shape[1]
    for i in range(len(target_id)):
        padding[:,padding_dim-len(target_id)+i] = target_id[i]
    if debug:
        print("input_ids is:",input_ids)
        print("target_id is:",target_id)
        print("padding is:",padding)
    return padding

def get_subset(frac ,dataset):
    if frac < 1.0:
        # Use a subset of the test dataset if frac < 1.0
        dataset_size = len(dataset)
        subset_size = int(frac * dataset_size)
        indices = np.arange(subset_size)
        dataset = torch.utils.data.Subset(dataset, indices)                
    return dataset

def find_next_run_dir(output_dir):
    base_dir = output_dir
    if not base_dir.endswith('/'):  # make sure the directory ends with '/'
        base_dir += '/'

    run_num = 1
    next_dir = base_dir + f"run_{run_num}"

    # Increment run_num until a non-existing directory is found
    while os.path.exists(next_dir) and len(os.listdir(next_dir)) > 0:
        run_num += 1
        next_dir = base_dir + f"run_{run_num}"
    
    return next_dir

#%%
def get_unique_test_image_ids():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    with open(f"{config['data_root']}/v2_mscoco_val2014_annotations.json",'r') as f:
        eval_file = json.load(f)
    annos = eval_file["annotations"]
    total_id = set([i["image_id"] for i in annos][:5000])
    print("total_id is:",len(total_id))
    print("total_id is:",total_id)
    return total_id

def get_unique_test_image_ids_over_15():
    total_id = np.load("data/multi_ques_15.npy")
    return list(total_id)

def create_id_question_map(image_ids: list[str], num: int, question_pool: list,path =None):
    
    query_path = f"data/question_list/num_{num}"
    
    # use speicified path if provided
    if path!=None:
        query_path = path
        
    # assert not os.path.exists(query_path), f"Question list for {num} questions per image already exists.
    os.makedirs(query_path, exist_ok=True)
    # create a dictionary to store the id-question pair
    id_to_ques = {}
    assert len(question_pool) > num, f"Question pool should be larger than {num}."
    # sample num questions from question pool for each image
    for id in image_ids:
        id_to_ques[str(id)] = random.sample(question_pool, num)
    
    # store the id-question pair in json file
    with open(f'{query_path}/id_to_question.json', 'w') as f:
        json.dump(id_to_ques, f)
    
    print(f"ID to question map is created at {query_path}/id_to_question.json")
  

def get_img_id_train_ques_map(num):
    
    query_path = f"data/question_list/num_{num}"
    with open(f'{query_path}/id_to_question.json', 'r') as f:
        id_ques_map = json.load(f)
    return id_ques_map

def get_img_id_train_prompt_map(num):
    query_path = f"data/prompt_list/num_{num}"
    with open(f'{query_path}/id_to_question.json', 'r') as f:
        id_ques_map = json.load(f)
    return id_ques_map

def get_ques(ids_ques_map,id,num,question_pool):
    if num == 0:
        return []
    
    file_path = f"data/question_list/num_{num}"
    single_id_ques_path = f"{file_path}/{id}.json"
    if id not in ids_ques_map:
        if os.path.exists(single_id_ques_path):
            with open(single_id_ques_path,'r') as f:
                id_ques_map = json.load(f)
                return id_ques_map[id]
        else:
            id_to_ques = {}    
            id_to_ques[id] = random.sample(question_pool, num)
            with open(single_id_ques_path,'w') as f:
                json.dump(id_to_ques,f)
                return id_to_ques[id]
    else:
        return ids_ques_map[id]

def create_num_train_ques_over_15(num = None):
        total_id = get_unique_test_image_ids_over_15()
        ques_pool = train_question()        
        create_id_question_map(total_id, num, ques_pool, path = f"data/question_list_over_15/num_{num}")
    
def load_datasets(args, load_mode = "both", dataset_name = "vqav2"):
    if dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path        
        test_questions_json_path = args.test_questions_json_path
        test_annotations_json_path = args.test_annotations_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    
    test_dataset = CroPADataset(
        image_dir_path=test_image_dir_path,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=dataset_name,
    )
    if load_mode == "test":
        return None, test_dataset
    
    train_dataset = CroPADataset(
        image_dir_path=train_image_dir_path,
        question_path=train_questions_json_path,
        annotations_path=train_annotations_json_path,
        is_train=True,
        dataset_name=dataset_name,
    )    
    return train_dataset, test_dataset

def load_instructblip_model(device,module):
    INSTRUCT_BLIP_PATH = "Salesforce/instructblip-vicuna-7b"    
    model_args = {
        'lm_path': INSTRUCT_BLIP_PATH,         
        'processor_path': INSTRUCT_BLIP_PATH,
        'device': f'{device}', 
    }
    eval_model = module.EvalModel(model_args)
    return eval_model

def load_blip_model(device,module):
    model_args = {
        'lm_path': 'Salesforce/blip2-opt-2.7b',         
        'processor_path': 'Salesforce/blip2-opt-2.7b',
        'device': f'{device}', 
    }
    eval_model = module.EvalModel(model_args)
    return eval_model

def load_flamingo_model(device,module):
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        
    model_args = {
        'lm_path': 'luodian/llama-7b-hf', 
        'lm_tokenizer_path': 'luodian/llama-7b-hf',
        'vision_encoder_path': 'ViT-L-14',
        'vision_encoder_pretrained': 'openai', 
        'checkpoint_path': config['flamingo_checkpoint_path'], 
        'cross_attn_every_n_layers': '4',
        'device': f'{device}', 
    }
    
    eval_model = module.EvalModel(model_args)
    return eval_model

def load_model(device,module,model_name):
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    login(token = config["hf_login_token"])
    print("model_name is:",model_name)
    
    if model_name=="blip2":
        return load_blip_model(device,module)
    elif model_name=="open_flamingo":
        return load_flamingo_model(device,module)
    elif model_name=="instructblip":
        return load_instructblip_model(device,module)
    else:
        raise ValueError("model name is not valid")

if __name__=="__main__":
    seed_everything(42)
    for i in [1,2,5,10,50,100]:
        create_num_train_ques_over_15(i)
        
    
#%%