#%%
import argparse
import importlib
import json
import os
import more_itertools
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Tuple
from utils.attack_tool import (
    add_extra_args,  get_available_gpus, get_img_id_train_prompt_map,
    get_subset, load_datasets, load_model, seed_everything
)
from utils.eval_model import BaseEvalModel
from utils.eval_tools import (
    get_eval_icl, load_icl_example, get_vqa_type,
    cap_instruction, cls_instruction, load_img_specific_questions, vqa_agnostic_instruction, 
    postprocess_generation,record_format_summary, record_format_summary_affect
)

#%%
if __name__=="__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prompt_num", type=int, default=10,
                        help="The number of prompts utilized during the optimization phase")
    parser.add_argument("--device", type=int, default=-1,
                        help="The device id of the GPU to use")
    parser.add_argument("--iter_num", type=int, default=300,
                        help="The num of attack iterations")
    parser.add_argument("--model_name", type=str, default="open_flamingo", #before: instructblip
                        help="The num of attack iter")
    parser.add_argument("--quick_eval", type=bool, default=False,
                        help="set to false to generate the result given clean images")
    parser.add_argument("--fraction", type=float, default=0.05,
                        help="The fraction of the test dataset to use")
    parser.add_argument("--shot", type=int, default=0,
                        help="The num of in context learning examples to use, specific for Flamingo")
    parser.add_argument("--method", type=str, default="cropa",
                        help="The mehod of attack, either cropa or baseline")
    parser.add_argument("--noise_dir", type=str, 
                        help="The directory of the attack noise")
    
    config_args = parser.parse_known_args()[0]
    assert config_args.method in ["cropa","baseline"], "method not supported"
    add_extra_args(config_args, config_args.model_name)
    
    module = importlib.import_module(f"open_flamingo.eval.models.{config_args.model_name}")
    if config_args.device >= 0:
        print("use specified gpu",config_args.device)
    else:
        config_args.device= get_available_gpus(45000)[0]
    device= f"cuda:{ config_args.device}"
    eval_model = load_model(config_args.device,module,config_args.model_name)
    train_dataset, test_dataset = load_datasets(config_args)
    num_shots = config_args.shot
    prompt_num = config_args.prompt_num

    if config_args.method == "baseline":    
        alpha2 = 0
    else:    
        prompt_num_to_alpha2 = config_args.prompt_num_to_alpha2    
        alpha2 = prompt_num_to_alpha2[prompt_num]
#%%       
def test(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    num_shots: int = 2,
    fraction: float = 0.01,
    target: str = "unknown<|endofchunk|>",
    noise_dir: str = "./",
    datasets: Tuple[Dataset, Dataset] = None,
):
    
    model_name = args.model_name        
    tokenizer = eval_model.tokenizer
    target = target.lower().strip().replace("_", " ")
    target_token_len = len(tokenizer.encode(target))-1
    print("target_token_len is:",target_token_len)
    train_dataset, test_dataset = datasets if datasets is not None else load_datasets(args=args)
    train_batch_demo_samples,test_batch_demo_samples = load_icl_example(train_dataset)
    test_dataset = get_subset(frac =  fraction, dataset = test_dataset)
                
    with open(args.vqav2_eval_annotations_json_path, "r") as f:
        eval_file =  json.load(f)
    annos = eval_file["annotations"]
    ques_id_to_img_id = {i["question_id"]:i["image_id"] for i in annos}    
    
    assert prompt_num >= 0, "require at least one question"
    img_id_to_train_prompt = get_img_id_train_prompt_map(prompt_num)
    
    # total_vqa_success_rate = []
    # total_cls_success_rate = []
    # total_cap_success_rate = []
    result_json = defaultdict(list)
    # loss_json = defaultdict(list)
    task_list = ["vqa","vqa_specific","cls","cap"]
    # image_set = set()
    
    vqa_specific_instruction = load_img_specific_questions() 
    
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    # with open("data/clean_train_vqa_map.json") as f:
    #     clean_vqa_model_output = json.load(f)
    
    tpoch = tqdm(test_dataset)
    for id,item in enumerate(tpoch):            

        img_id = str(ques_id_to_img_id[item["question_id"]])       
        item_images = []
        item_text = []    
        total_prompt_list  = img_id_to_train_prompt[img_id]
        print("total_ques_list size is:",len(total_prompt_list))
        
        if num_shots > 0:
            print("batch_demo_samples is:",train_batch_demo_samples)
            context_images = [x["image"] for x in train_batch_demo_samples]
        else:
            context_images = []
            
        item_images.append(context_images + [item["image"]])
        print("item_images is:",item_images)
        
        if num_shots > 0:
            test_item_images = [[x["image"] for x in test_batch_demo_samples]+[item_images[0][-1]]]
        else:
            test_item_images = [[item_images[0][-1]]]
        
        print("test_item_images is:",test_item_images)
        train_context_text = "".join([
                eval_model.get_vqa_prompt(
                    question=x["question"], answer=x["answers"][0]
                )
                for x in train_batch_demo_samples
        ])

        if num_shots == 0:
            train_context_text = train_context_text.replace("<image>", "")
         
        if model_name in ["blip2","instructblip"]:
            train_context_text=""

        for ques in total_prompt_list:
            item_text.append(
                train_context_text + eval_model.get_vqa_prompt(question=ques)+" "+target)
            
                   
        attack  =  np.load(f"{noise_dir}/{ques_id_to_img_id[item['question_id']]}_.npy")
        attack = torch.from_numpy(attack).to(device) 
        vqa_agnostic_sample = vqa_agnostic_instruction()
        vqa_specific_sample = vqa_specific_instruction[img_id]
        prompt_list = [vqa_agnostic_sample,vqa_specific_sample[:10],cls_instruction(),cap_instruction()]
        vqa_stats  = {"number":{"success":0,"total":0},
                    "yes_no":{"success":0,"total":0},
                    "what":{"success":0,"total":0},
                    "where":{"success":0,"total":0},
                    "other":{"success":0,"total":0}}
        
        template_list = [eval_model.get_vqa_prompt,eval_model.get_vqa_prompt,eval_model.get_classification_prompt,eval_model.get_caption_prompt]
        # result_list = [total_vqa_success_rate,total_cls_success_rate,total_cap_success_rate]
        for  i in range(len(prompt_list)):
            task_name = task_list[i]
            instruction_list = prompt_list[i]
            template_func = template_list[i]
            
            success_count = 0        
            target_success_count = 0           
            test_context_text = get_eval_icl(task_name,num_shots, test_batch_demo_samples,eval_model)            
            for batch_ques in more_itertools.chunked(instruction_list,args.eval_batch_size):
                
                if task_name == "vqa" or task_name=="vqa_specific":
                    eval_text = [test_context_text+template_func(ques) for ques in batch_ques]
                else:           
                    eval_text = [test_context_text+"<image>"+instruction+" Output:" for instruction in batch_ques]
                # delete any in-context prompt
                if model_name in ["blip2","instructblip"]:
                    eval_text = ["Context:"+instruction+" Answer:" for instruction in batch_ques]
                    
                if model_name=="instructblip":
                    test_item_images=item_images[0]
                
                
                outputs = eval_model.get_outputs_attack(
                    attack = attack,batch_images=test_item_images*len(batch_ques),
                    batch_text=eval_text,max_generation_length=max_generation_length,
                    num_beams=num_beams,length_penalty=length_penalty)                        
                if not args.quick_eval:
                    clean_outputs = eval_model.get_outputs(
                        batch_images=test_item_images*len(batch_ques),
                        batch_text=eval_text,max_generation_length=max_generation_length,
                        num_beams=num_beams,length_penalty=length_penalty)
                
                process_function = postprocess_generation
                new_predictions = list(map(process_function, outputs))
                clean_newpredictions = list(map(process_function, clean_outputs)) if not args.quick_eval else None
                for i in range(len(new_predictions)):
                    target_attack_is_success = False
                    if clean_newpredictions is not None and new_predictions[i]!=clean_newpredictions[i]:
                        success_count+=1    
                        
                    if new_predictions[i].strip().lower() ==target.lower().split("<")[0].strip():
                        target_success_count+=1 
                        target_attack_is_success = True
                    
                    if task_name == "vqa" or task_name=="vqa_specific":
                        prompt_type = get_vqa_type(batch_ques[i])
                        if target_attack_is_success:vqa_stats[prompt_type]["success"]+=1 
                        vqa_stats[prompt_type]["total"]+=1
                                            
            print("success_count is:",success_count,"**target_success_count is**:",target_success_count,item["question"])
            with open(f"{output_dir}/results.txt","a") as f:
                f.write(f"{task_name}, target num:{target_success_count}, total attack num:{success_count}\n")
            result_json[task_name].append({"count":success_count,"target_count":target_success_count})            
        
        for i in vqa_stats.keys():
            # vqa_stats[i] = {"success_rate":vqa_stats[i]["success"]/vqa_stats[i]["total"],"total":vqa_stats[i]["total"]}
            result_json[i].append(vqa_stats[i])             
        print(f"result of {task_name}.is:",result_json[task_name])           
    
    result_summary = {}
    mean_res = {}    
    for t_name in task_list:
        mean_success_count = np.mean([i["count"] for i in result_json[t_name]]) if not args.quick_eval else -1
        mean_target_success_count = np.mean([i["target_count"] for i in  result_json[t_name]])
        
        if t_name == "vqa" or t_name == "vqa_specific"  :
            rate = "{:.4f}".format(mean_target_success_count/10)
            affect_rate = "{:.4f}".format(mean_success_count/10) 
        else:
            rate = "{:.4f}".format(mean_target_success_count/20)
            affect_rate = "{:.4f}".format(mean_success_count/20)             
        mean_res[t_name]={
            "target_rate":rate,
            "mean_count":mean_success_count,
            "mean_target_count":mean_target_success_count,
            "mean_affect_rate":affect_rate}
    mean_vqa_stats = {}
    
    for i in vqa_stats.keys():
        splited_success = np.mean([i["success"] for i in result_json[i]])
        mean_total_num = np.mean([i["total"] for i in result_json[i]])
        mean_vqa_stats[i] = {"mean_success":splited_success,
                            "mean_total_num":mean_total_num,
                            "mean_success_rate":"{:.2f}".format(splited_success/mean_total_num)}
    result_json["avg"] = mean_res
    result_json["vqa_stats"] = mean_vqa_stats
    print(result_json)
    json.dump(result_json,open(f"{output_dir}/total_success_rate.json","w"))      
    record_format_summary(result_summary,output_dir) 
    if not args.quick_eval:
        record_format_summary_affect(result_summary,output_dir) 
    
target_text = "unknown" 
# test(
#     config_args,
#     eval_model = eval_model,
#     max_generation_length = 5,
#     num_beams= 3,
#     length_penalty = -2.0,
#     num_shots = num_shots ,
#     alpha1 = 1/255,
#     epsilon = 16/255,
#     fraction=config_args.fraction,
#     target = target_text+config_args.eoc,
#     base_dir = f"output/{config_args.model_name}_shots_{num_shots}/{config_args.method}/num_{prompt_num}_{target_text}",
#     alpha2 = alpha2 ,
#     prompt_num=config_args.prompt_num,
#     datasets=(train_dataset,  test_dataset),
# )
test(
    args = config_args, 
    eval_model = eval_model,
    max_generation_length = 5,
    num_beams = 3,
    length_penalty = -2.0,
    num_shots = 2,
    fraction = 0.002,
    target = "unknown<|endofchunk|>",
    noise_dir = "/homes/55/haochen/openflamingo/open_flamingo/eval/open_flamingo_15/simt/alter/1501_16_255_ep_alpha_1_255_ques_10_unknown_0.01/oneImgMultiQues_0.05/run_2/1300",
    datasets = (train_dataset, test_dataset)
)




    