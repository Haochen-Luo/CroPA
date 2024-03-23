#%%
import json
import os,re
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
#%%
def plot_loss(losses:list,save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training loss")
    plt.title("Training Loss over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/loss.pdf")


def approximate_embeddings(input_tensor, model,tokenizer, top_k=3):
    with torch.no_grad():
        # Normalize input tensor
        input_tensor = input_tensor / input_tensor.norm(dim=-1, keepdim=True)

        # Get the embeddings from the model
        embeddings = model.get_input_embeddings().weight
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # Calculate the cosine similarity between your input tensor and all embeddings
        cos_sim = torch.matmul(input_tensor, embeddings.T)

        # Get the indices of the top k most similar embeddings
        _, top_k_indices = torch.topk(cos_sim, top_k, dim=-1)

        # Decode the top k most similar tokens
        tokens = [[[tokenizer.decode([idx.item()]) for idx in item] for item in row] for row in top_k_indices]
    return tokens

def compute_effective_num_shots(num_shots, model_type):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots

def get_testset(frac ,dataset):
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

def record_format_summary(file,path):
    with open(os.path.join(path,"result.txt"),'w') as f2:
        f2.write('    vqa,vqa_sp,cls,cap\n')
        if 'avg' in file.keys():
            res = file['avg']
            f2.write("avg,"+",".join([i['target_rate'] for i in res.values()])+"\n")
        # otherwise, it requires unpacking thorugh epoch
        else:
            for key in file.keys():
                res = file[key]['avg']
                f2.write(str(key)+","+",".join([i['target_rate'] for i in res.values()])+"\n")


def record_format_summary_affect(file,path):
    with open(os.path.join(path,"result_affect.txt"),'w') as f2:
        f2.write('    vqa,vqa_sp,cls,cap\n')
        for key in file.keys():
            res = file[key]['avg']
            f2.write(str(key)+","+",".join([i["mean_affect_rate"] for i in res.values()])+"\n")

def format_summary(path):
    with open(f"{path}", "r") as f:    
        file = json.load(f)
        print('    vqa,vqa_sp,cls,cap')
        for key in file.keys():
            
            res = file[key]['avg']
            print(str(key)+","+",".join([i['target_rate'] for i in res.values()]))
           
def cls_instruction():
    """
    Generate Prompts: Suppose you need to write prompt to let vision-language model to do the image classification task, can you write some prompt. Please write 20 in python code block. Please do not assume the content of the image, write generic ones. Your prompts should be varied.
    """   
    instructions = [
        "Imagine you're explaining this image to someone over the phone. Please describe the image in one word?"
        "Perform the image classification task on this image. Give the label in one word.",
        "Imagine a child is trying to identify the image. What might they excitedly point to and name?",        
        "If this image were turned into a jigsaw puzzle, what would the box label say to describe the picture inside?",
        "Classify the content of this image.",
        "If you were to label this image, what label would you give?",
        "What category best describes this image?",
        "Describe the central subject of this image in a single word.",
        "Provide a classification for the object depicted in this image.",
        "If this image were in a photo album, what would its label be?",
        "Categorize the content of the image.",
        "If you were to sort this image into a category, which one would it be?",
        "What keyword would you associate with this image?",
        "Assign a relevant classification to this image.",
        "If this image were in a gallery, under which section would it belong?",
        "Describe the main theme of this image in one word.",
        "Under which category would this image be cataloged in a library?",
        "What classification tag fits this image the best?",
        "Provide a one-word description of this image's content.",
        "If you were to archive this image, what descriptor would you use?"
    ]

    return instructions

def full_cls_instruction():
    more_cls_instructions = [
    "Identify the primary theme of this image in one word.",
    "How would you label this image with a single descriptor?",
    "Determine the main category for this image.",
    "Offer a one-word identifier for this picture.",
    "If this image were a file on your computer, what would its name be?",
    "Tag this image with its most relevant keyword.",
    "Provide the primary classification for this photograph.",
    "How would you succinctly categorize this image?",
    "Offer the primary descriptor for the content of this image.",
    "If this image were a product, what label would you place on its box?",
    "Choose a single word that encapsulates the image's content.",
    "How would you classify this image in a database?",
    "In one word, describe the essence of this image.",
    "Provide the most fitting category for this image.",
    "What is the principal subject of this image?",
    "If this image were in a store, which aisle would it belong to?",
    "Provide a singular term that characterizes this picture.",
    "How would you caption this image in a photo contest?",
    "Select a label that fits the main theme of this image.",
    "Offer the most appropriate tag for this image.",
    "Which keyword best summarizes this image?",
    "How would you title this image in an exhibition?",
    "Provide a succinct identifier for the image's content.",
    "Choose a word that best groups this image with others like it.",
    "If this image were in a museum, how would it be labeled?",
    "Assign a central theme to this image in one word.",
    "Tag this photograph with its primary descriptor.",
    "What is the overriding theme of this picture?",
    "Provide a classification term for this image.",
    "How would you sort this image in a collection?",
    "Identify the main subject of this image concisely.",
    "If this image were a magazine cover, what would its title be?",
    "What term would you use to catalog this image?",
    "Classify this picture with a singular term.",
    "If this image were a chapter in a book, what would its title be?",
    "Select the most fitting classification for this image.",
    "Define the essence of this image in one word.",
    "How would you label this image for easy retrieval?",
    "Determine the core theme of this photograph.",
    "In a word, encapsulate the main subject of this image.",
    "If this image were an art piece, how would it be labeled in a gallery?",
    "Provide the most concise descriptor for this picture.",
    "How would you name this image in a photo archive?",
    "Choose a word that defines the image's main content.",
    "What would be the header for this image in a catalog?",
    "Classify the primary essence of this picture.",
    "What label would best fit this image in a slideshow?",
    "Determine the dominant category for this photograph.",
    "Offer the core descriptor for this image.",
    "If this image were in a textbook, how would it be labeled in the index?",
    "Select the keyword that best defines this image's theme.",
    "Provide a classification label for this image.",
    "If this image were a song title, what would it be?",
    "Identify the main genre of this picture.",
    "Assign the most apt category to this image.",
    "Describe the overarching theme of this image in one word.",
    "What descriptor would you use for this image in a portfolio?",
    "Summarize the image's content with a single identifier."
        
    ]
    return more_cls_instructions+cls_instruction()

def cap_instruction():
    instructions = [
        "Describe the content of this image.",
        "What's happening in this image?",
        "Provide a brief caption for this image.",
        "Tell a story about this image in one sentence.",
        "If this image could speak, what would it say?",
        "Summarize the scenario depicted in this image.",
        "What is the central theme or event shown in the picture?",
        "Create a headline for this image.",
        "Explain the scene captured in this image.",
        "If this were a postcard, what message would it convey?",
        "Narrate the visual elements present in this image.",
        "Give a short title to this image.",
        "How would you describe this image to someone who can't see it?",
        "Detail the primary action or subject in the photo.",
        "If this image were the cover of a book, what would its title be?",
        "Translate the emotion or event of this image into words.",
        "Compose a one-liner describing this image's content.",
        "Imagine this image in a magazine. What caption would go with it?",
        "Capture the essence of this image in a brief description.",
        "Narrate the visual story displayed in this photograph."
    ]
    return instructions

def full_cap_instruction():
    more_instructions = [
         "Elaborate on the elements present in this image.",
        "In one sentence, summarize the activity in this image.",
        "Relate the main components of this picture in words.",
        "What narrative unfolds in this image?",
        "Break down the main subjects of this photo.",
        "Give an account of the main scene in this image.",
        "In a few words, state what this image represents.",
        "Describe the setting or location captured in this photograph.",
        "Provide an overview of the subjects or objects seen in this picture.",
        "Identify the primary focus or point of interest in this image.",
        "What would be the perfect title for this image?",
        "How would you introduce this image in a presentation?",
        "Present a quick rundown of the image's main subject.",
        "What's the key event or subject captured in this photograph?",
        "Relate the actions or events taking place in this image.",
        "Convey the content of this photograph in a single phrase.",
        "Offer a succinct description of this picture.",
        "Give a concise overview of this image.",
        "Translate the contents of this picture into a sentence.",
        "Describe the characters or subjects seen in this image.",
        "Capture the activities happening in this image with words.",
        "How would you introduce this image to an audience?",
        "State the primary events or subjects in this picture.",
        "What are the main elements in this photograph?",
        "Provide an interpretation of this image's main event or subject.",
        "How would you title this image for an art gallery?",
        "What scenario or setting is depicted in this image?",
        "Concisely state the main actions occurring in this image.",
        "Offer a short summary of this photograph's contents.",
        "How would you annotate this image in an album?",
        "If you were to describe this image on the radio, how would you do it?",
        "In your own words, narrate the main event in this image.",
        "What are the notable features of this image?",
        "Break down the story this image is trying to tell.",
        "Describe the environment or backdrop in this photograph.",
        "How would you label this image in a catalog?",
        "Convey the main theme of this picture succinctly.",
        "Characterize the primary event or action in this image.",
        "Provide a concise depiction of this photo's content.",
        "Write a brief overview of what's taking place in this image.",
        "Illustrate the main theme of this image with words.",
        "How would you describe this image in a gallery exhibit?",
        "Highlight the central subjects or actions in this image.",
        "Offer a brief narrative of the events in this photograph.",
        "Translate the activities in this image into a brief sentence.",
        "Give a quick rundown of the primary subjects in this image.",
        "Provide a quick summary of the scene captured in this photo.",
        "How would you explain this image to a child?",
        "What are the dominant subjects or objects in this photograph?",
        "Summarize the main events or actions in this image.",
        "Describe the context or setting of this image briefly.",
        "Offer a short description of the subjects present in this image.",
        "Detail the main scenario or setting seen in this picture.",
        "Describe the main activities or events unfolding in this image.",
        "Provide a concise explanation of the content in this image.",
        "If this image were in a textbook, how would it be captioned?",
        "Provide a summary of the primary focus of this image.",
        "State the narrative or story portrayed in this picture.",
        "How would you introduce this image in a documentary?",
        "Detail the subjects or events captured in this image.",
        "Offer a brief account of the scenario depicted in this photograph.",
        "State the main elements present in this image concisely.",
        "Describe the actions or events happening in this picture.",
        "Provide a snapshot description of this image's content.",
        "How would you briefly describe this image's main subject or event?"
    ]
    return more_instructions+cap_instruction()
    
def vqa_agnostic_instruction():

    total_instructions = agnostic_question()
    instructions = random.sample(total_instructions, 10)
    return instructions


def load_img_specific_questions(): 
    with open("data/img_specific_questions.json","r") as f:
        specific_questions = json.load(f)
    return specific_questions

def agnostic_question():
    questions = [
        'Any cutlery items visible in the image?',
        'Any bicycles visible in this image?',
        'Any boats visible in the image?',
        'Any bottles present in the image?',
        'Are curtains noticeable in the image?',
        'Are flags present in the image?',
        'Are flowers present in the image?',
        'Are fruits present in the image?',
        'Are glasses discernible in the image?',
        'Are hills visible in the image?',
        'Are plates discernible in the image?',
        'Are shoes visible in this image?',
        'Are there any insects in the image?',
        'Are there any ladders in the image?',
        'Are there any man-made structures in the image?',
        'Are there any signs or markings in the image?',
        'Are there any street signs in the image?',
        'Are there balloons in the image?',
        'Are there bridges in the image?',
        'Are there musical notes in the image?',
        'Are there people sitting in the image?',
        'Are there skyscrapers in the image?',
        'Are there toys in the image?',
        'Are toys present in this image?',
        'Are umbrellas discernible in the image?',
        'Are windows visible in the image?',
        'Can birds be seen in this image?',
        'Can stars be seen in this image?',
        'Can we find any bags in this image?',
        'Can you find a crowd in the image?',
        'Can you find a hat in the image?',
        'Can you find any musical instruments in this image?',
        'Can you identify a clock in this image?',
        'Can you identify a computer in this image?',
        'Can you see a beach in the image?',
        'Can you see a bus in the image?',
        'Can you see a mailbox in the image?',
        'Can you see a mountain in the image?',
        'Can you see a staircase in the image?',
        'Can you see a stove or oven in the image?',
        'Can you see a sunset in the image?',
        'Can you see any cups or mugs in the image?',
        'Can you see any jewelry in the image?',
        'Can you see shadows in the image?',
        'Can you see the sky in the image?',
        'Can you spot a candle in this image?',
        'Can you spot a farm in this image?',
        'Can you spot a pair of shoes in the image?',
        'Can you spot a rug or carpet in the image?',
        'Can you spot any dogs in the image?',
        'Can you spot any snow in the image?',
        'Do you notice a bicycle in the image?',
        'Does a ball feature in this image?',
        'Does a bridge appear in the image?',
        'Does a cat appear in the image?',
        'Does a fence appear in the image?',
        'Does a fire feature in this image?',
        'Does a mirror feature in this image?',
        'Does a table feature in this image?',
        'Does it appear to be nighttime in the image?',
        'Does it look like an outdoor image?',
        'Does it seem to be countryside in the image?',
        'Does the image appear to be a cartoon or comic strip?',
        'Does the image contain any books?',
        'Does the image contain any electronic devices?',
        'Does the image depict a road?',
        'Does the image display a river?',
        'Does the image display any towers?',
        'Does the image feature any art pieces?',
        'Does the image have a lamp?',
        'Does the image have any pillows?',
        'Does the image have any vehicles?',
        'Does the image have furniture?',
        'Does the image primarily display natural elements?',
        'Does the image seem like it was taken during the day?',
        'Does the image seem to be taken indoors?',
        'Does the image show any airplanes?',
        'Does the image show any benches?',
        'Does the image show any landscapes?',
        'Does the image show any movement?',
        'Does the image show any sculptures?',
        'Does the image show any signs?',
        'Does the image show food?',
        'Does the image showcase a building?',
        'How many animals are present in the image?',
        'How many bikes are present in the image?',
        'How many birds are visible in the image?',
        'How many buildings can be identified in the image?',
        'How many cars can be seen in the image?',
        'How many doors can you spot in the image?',
        'How many flowers can be identified in the image?',
        'How many trees feature in the image?',
        'Is a chair noticeable in the image?',
        'Is a computer visible in the image?',
        'Is a forest noticeable in the image?',
        'Is a painting visible in the image?',
        'Is a path or trail visible in the image?',
        'Is a phone discernible in the image?',
        'Is a train noticeable in the image?',
        'Is sand visible in the image?',
        'Is the image displaying any clouds?',
        'Is the image set in a city environment?',
        'Is there a plant in the image?',
        'Is there a source of light visible in the image?',
        'Is there a television displayed in the image?',
        'Is there grass in the image?',
        'Is there text in the image?',
        'Is water visible in the image, like a sea, lake, or river?',
        "How many people are captured in the image?",
        "How many windows can you count in the image?",
        "How many animals, other than birds, are present?",
        "How many statues or monuments stand prominently in the scene?",
        "How many streetlights are visible?",
        "How many items of clothing can you identify?",
        "How many shoes can be seen in the image?",
        "How many clouds appear in the sky?",
        "How many pathways or trails are evident?",
        "How many bridges can you spot?",
        "How many boats are present, if it's a waterscape?",
        "How many pieces of fruit can you identify?",
        "How many hats are being worn by people?",
        "How many different textures can you discern?",
        "How many signs or billboards are visible?",
        "How many musical instruments can be seen?",
        "How many flags are present in the image?",
        "How many mountains or hills can you identify?",
        "How many books are visible, if any?",
        "How many bodies of water, like ponds or pools, are in the scene?",
        "How many shadows can you spot?",
        "How many handheld devices, like phones, are present?",
        "How many pieces of jewelry can be identified?",
        "How many reflections, perhaps in mirrors or water, are evident?",
        "How many pieces of artwork or sculptures can you see?",
        "How many staircases or steps are in the image?",
        "How many archways or tunnels can be counted?",
        "How many tools or equipment are visible?",
        "How many modes of transportation, other than cars and bikes, can you spot?",
        "How many lamp posts or light sources are there?",
        "How many plants, other than trees and flowers, feature in the scene?",
        "How many fences or barriers can be seen?",
        "How many chairs or seating arrangements can you identify?",
        "How many different patterns or motifs are evident in clothing or objects?",
        "How many dishes or food items are visible on a table setting?",
        "How many glasses or mugs can you spot?",
        "How many pets or domestic animals are in the scene?",
        "How many electronic gadgets can be counted?",
        "Where is the brightest point in the image?",
        "Where are the darkest areas located?",
        "Where can one find leading lines directing the viewer's eyes?",
        "Where is the visual center of gravity in the image?",
        "Where are the primary and secondary subjects positioned?",
        "Where do the most vibrant colors appear?",
        "Where is the most contrasting part of the image located?",
        "Where does the image place emphasis through scale or size?",
        "Where do the textures in the image change or transition?",
        "Where does the image break traditional compositional rules?",
        "Where do you see repetition or patterns emerging?",
        "Where does the image exhibit depth or layers?",
        "Where are the boundary lines or borders in the image?",
        "Where do different elements in the image intersect or overlap?",
        "Where does the image hint at motion or movement?",
        "Where are the calm or restful areas of the image?",
        "Where does the image become abstract or less defined?",
        "Where do you see reflections, be it in water, glass, or other surfaces?",
        "Where does the image provide contextual clues about its setting?",
        "Where are the most detailed parts of the image?",
        "Where do you see shadows, and how do they impact the composition?",
        "Where can you identify different geometric shapes?",
        "Where does the image appear to have been cropped or framed intentionally?",
        "Where do you see harmony or unity among the elements?",
        "Where are there disruptions or interruptions in patterns?",
        "What is the spacing between objects or subjects in the image?",
        "What foreground, mid-ground, and background elements can be differentiated?",
        "What type of energy or vibe does the image exude?",
        "What might be the sound environment based on the image's content?",
        "What abstract ideas or concepts does the image seem to touch upon?",
        "What is the relationship between the main subjects in the image?",
        "What items in the image could be considered rare or unique?",
        "What is the gradient or transition of colors like in the image?",
        "What might be the smell or aroma based on the image's content?",
        "What type of textures can be felt if one could touch the image's content?",
        "What boundaries or limits are depicted in the image?",
        "What is the socioeconomic context implied by the image?",
        "What might be the immediate aftermath of the scene in the image?",
        "What seems to be the main source of tension or harmony in the image?",
        "What might be the narrative or backstory of the main subject?",
        "What elements of the image give it its primary visual weight?",
        'Would you describe the image as bright or dark?',
        'Would you describe the image as colorful or dull?',
    ]

    return questions

def load_ques_type_dict(ques):
    type_map = {}
    for i in ques:
        if i.lower().startswith("how many"):
            type_map[i] = "number"
        elif i.lower().startswith("where"):
            type_map[i] = "where"
        elif i.lower().startswith("what"):
            type_map[i] = "what"
        else:
            type_map[i] = "yes_no"
    return type_map

def get_vqa_type(i):
    i_lower = i.lower()
    if i_lower.startswith("how many"):
        return "number"
    elif i_lower.startswith("where"):
        return "where"
    elif i_lower.startswith("what"):
        return "what"
    elif i_lower.startswith(("is", "are", "will", "can", "do", "does", "has", "have", "did", "were", "was", "should","any")):
        return "yes_no"
    else:
        return "other"
    
def train_question_no_leakage():
    train_questions = [
        "Are there any annotations on the image?",
        "Are there any bodies of water other than seas, lakes, or rivers in the image?",
        "Are there any buildings that aren't skyscrapers in the image?",
        "Are there any filters applied to the image?",
        "Are there any furniture items other than chairs or tables in the image?",
        "Are there any geometric shapes in the image?",
        "Are there any hidden elements in the image?",
        "Are there any particles in the image like rain, snow, or dust?",
        "Are there any paths or roads in the image that are not main roads?",
        "Are there any patterns or textures in the image?",
        "Are there any recognizable brands in the image?",
        "Are there any reflections in the image?",
        "Are there any repeated elements in the image?",
        "Are there any trees in the image, not including forests?",
        "Are there any unusual elements in the image?",
        "Are there any vehicles other than cars or bikes in the image?",
        "Does the image contain a horizon line?",
        "Does the image contain any celebrities?",
        "Does the image contain any controversial elements?",
        "Does the image contain any drawings or sketches?",
        "Does the image contain any fashion accessories?",
        "Does the image contain any food items not included in the main meal?",
        "Does the image contain any monuments or landmarks?",
        "Does the image contain any optical illusions?",
        "Does the image contain any symbols?",
        "Does the image depict any professional activities?",
        "Does the image depict any seasonal elements?",
        "Does the image depict night or day?",
        "Does the image have a frame?",
        "Does the image have a shallow or a deep depth of field?",
        "Does the image have alt text?",
        "Does the image have an associated hashtag?",
        "Does the image have any connotations or symbolism?",
        "Does the image have borders?",
        "Does the image have cultural significance?",
        "Does the image include a depth map?",
        "Does the image involve any wildlife?",
        "Does the image look vintage or modern?",
        "Does the image represent a special event or holiday?",
        "Does the image tell a story?",
        "Is it a photo, drawing, or painting?",
        "Is the image a close-up or long shot?",
        "Is the image a composite of multiple images?",
        "Is the image a long exposure shot?",
        "Is the image a meme?",
        "Is the image a panorama?",
        "Is the image a portrait or a group photo?",
        "Is the image a selfie?",
        "Is the image blurred or sharp?",
        "Is the image historical or contemporary?",
        "Is the image in a raw or compressed format?",
        "Is the image in black and white or color?",
        "Is the image part of a series or collection?",
        "Is the image part of an advertisement?",
        "Is the image photoshopped or digitally altered?",
        "Is the image shot from a drone or high viewpoint?",
        "Is the image staged or candid?",
        "Is there a signature in the image?",
        "Is there a title for the image?",
        "Is there an obvious use of rule of thirds in the image?",
        "Is there any motion blur in the image?",
        "Is there any use of symmetry in the image?",
        "What are the dimensions of the image?",
        "What are the dimensions of the main object in the image?",
        "What are the image's copyright information?",
        "What are the metadata of the image?",
        "What artistic techniques are used in the image?",
        "What cultural references can be found in the image?",
        "What direction is the light coming from in the image?",
        "What emotions does the image evoke?",
        "What historical events does the image reference?",
        "What is the age of the people in the image?",
        "What is the aspect ratio of the image?",
        "What elements in the image seem out of place or unexpected?",        # modified     "What is the caption of the image?"
        "What is the depth of field in the image?",
        "What is the dominant color in the image?",
        "What is the dominant material in the image?",
        "What is the focus of the image?",
        "What is the genre of the image?",
        "What kind of lighting is used in the image?", #modified
        "What patterns or repetitions can be observed in the image?",#modified
        "What interaction, if any, can be noticed between different elements in the image?",
        "What is the mood of the image?",
        "What is the orientation of the image? Portrait or landscape?",
        "What is the point of view in the image?",
        "What is the setting of the image?",
        "What is the source of light in the image?",
        "What is the style of the image?",
        "What is the subject's pose in the image?",
        "What is the texture  in the image?", #modified
        "What is the theme of the image?",
        "What is the weather like in the image?",
        "What kind of camera was used to take the photo?",
        "What kind of landscape does the image show, if any?",
        "What language is used in the image's text, if any?",
        "What size is the image file?",
        "What social issues does the image address?",
        "What time of day does the image depict?",
        "What type of animals are in the image, if any?",
        "What type of clothes are the people in the image wearing?",
        "What type of hairstyle do the people in the image have?",
    ]
    return train_questions

def train_question():
    train_questions = [
        "Are there any annotations on the image?",
        "Are there any bodies of water other than seas, lakes, or rivers in the image?",
        "Are there any buildings that aren't skyscrapers in the image?",
        "Are there any filters applied to the image?",
        "Are there any furniture items other than chairs or tables in the image?",
        "Are there any geometric shapes in the image?",
        "Are there any hidden elements in the image?",
        "Are there any particles in the image like rain, snow, or dust?",
        "Are there any paths or roads in the image that are not main roads?",
        "Are there any patterns or textures in the image?",
        "Are there any recognizable brands in the image?",
        "Are there any reflections in the image?",
        "Are there any repeated elements in the image?",
        "Are there any trees in the image, not including forests?",
        "Are there any unusual elements in the image?",
        "Are there any vehicles other than cars or bikes in the image?",
        "Does the image contain a horizon line?",
        "Does the image contain any celebrities?",
        "Does the image contain any controversial elements?",
        "Does the image contain any drawings or sketches?",
        "Does the image contain any fashion accessories?",
        "Does the image contain any food items not included in the main meal?",
        "Does the image contain any monuments or landmarks?",
        "Does the image contain any optical illusions?",
        "Does the image contain any symbols?",
        "Does the image depict any professional activities?",
        "Does the image depict any seasonal elements?",
        "Does the image depict night or day?",
        "Does the image have a frame?",
        "Does the image have a shallow or a deep depth of field?",
        "Does the image have alt text?",
        "Does the image have an associated hashtag?",
        "Does the image have any connotations or symbolism?",
        "Does the image have borders?",
        "Does the image have cultural significance?",
        "Does the image include a depth map?",
        "Does the image involve any wildlife?",
        "Does the image look vintage or modern?",
        "Does the image represent a special event or holiday?",
        "Does the image tell a story?",
        "Is it a photo, drawing, or painting?",
        "Is the image a close-up or long shot?",
        "Is the image a composite of multiple images?",
        "Is the image a long exposure shot?",
        "Is the image a meme?",
        "Is the image a panorama?",
        "Is the image a portrait or a group photo?",
        "Is the image a selfie?",
        "Is the image blurred or sharp?",
        "Is the image historical or contemporary?",
        "Is the image in a raw or compressed format?",
        "Is the image in black and white or color?",
        "Is the image part of a series or collection?",
        "Is the image part of an advertisement?",
        "Is the image photoshopped or digitally altered?",
        "Is the image shot from a drone or high viewpoint?",
        "Is the image staged or candid?",
        "Is there a signature in the image?",
        "Is there a title for the image?",
        "Is there an obvious use of rule of thirds in the image?",
        "Is there any motion blur in the image?",
        "Is there any use of symmetry in the image?",
        "What are the dimensions of the image?",
        "What are the dimensions of the main object in the image?",
        "What are the image's copyright information?",
        "What are the metadata of the image?",
        "What artistic techniques are used in the image?",
        "What cultural references can be found in the image?",
        "What direction is the light coming from in the image?",
        "What emotions does the image evoke?",
        "What historical events does the image reference?",
        "What is the age of the people in the image?",
        "What is the aspect ratio of the image?",
        "What is the caption of the image?",        
        "What is the depth of field in the image?",
        "What is the dominant color in the image?",
        "What is the dominant material in the image?",
        "What is the focus of the image?",
        "What is the genre of the image?",
        "What is the location where the image was taken?",
        "What is the main color palette of the image?",
        "What is the main object in the image?",
        "What is the mood of the image?",
        "What is the orientation of the image? Portrait or landscape?",
        "What is the point of view in the image?",
        "What is the setting of the image?",
        "What is the source of light in the image?",
        "What is the style of the image?",
        "What is the subject's pose in the image?",
        "What is the texture of the main object in the image?", 
        "What is the theme of the image?",
        "What is the weather like in the image?",
        "What kind of camera was used to take the photo?",
        "What kind of landscape does the image show, if any?",
        "What language is used in the image's text, if any?",
        "What size is the image file?",
        "What social issues does the image address?",
        "What time of day does the image depict?",
        "What type of animals are in the image, if any?",
        "What type of clothes are the people in the image wearing?",
        "What type of hairstyle do the people in the image have?",
    ]
    return train_questions

def fixed_cap_icl(eval_model):
    captions = [
        "the group of people are boarding the bus.",
        "a couple of people standing under a umbrella on a street."
    ]
    icl_text = [eval_model.get_caption_prompt(i) for i in captions]
    return "".join(icl_text)


def fixed_cls_icl(eval_model):
    
    class_names = [
        "bus",
        "umbrella"
    ]
    icl_text = [eval_model.get_classification_prompt(i) for i in class_names]
    
    return "".join(icl_text)

def get_eval_icl(task_name,num_shots, test_batch_demo_samples,eval_model):
    if task_name=="vqa" or task_name=="vqa_specific":
        test_context_text = "".join([
                eval_model.get_vqa_prompt(
                    question=x["question"], answer=x["answers"][0]
                )
                for x in test_batch_demo_samples
            ])
    elif task_name=="cap":
        test_context_text = fixed_cap_icl(eval_model)
    elif task_name=="cls":
        test_context_text = fixed_cls_icl(eval_model)
    else:
        raise NotImplementedError
        
    if num_shots == 0:
        test_context_text = test_context_text.replace("<image>", "")
    
    print("test_context_text is:",test_context_text)
    return test_context_text 

def postprocess_generation(predictions):
    if "Question" not in predictions and "Answer" not in predictions and "Short" not in predictions and "Long" not in predictions:
        return predictions
    answer = re.split("Question|Answer|Short|Long", predictions, 1)[0]
    answer = re.split(", |\.\s", answer, 1)[0]
    return answer

def load_icl_example(train_dataset,effective_num_shots = 2):
    """
    indices = np.random.choice(len(in_context_samples),2* effective_num_shots, replace=False) replace=False for unique elements
    train_indices = indices[:effective_num_shots ]
    test_indices = indices[effective_num_shots :2*effective_num_shots ]
    print("train_indices is:",train_indices,"test_indices is:",test_indices)
    """
       
    with open('data/icl_indices.json', 'r') as f:
        indices_map = json.load(f)
    train_indices = indices_map[str(effective_num_shots)]["train"]
    test_indices = indices_map[str(effective_num_shots)]["test"]        
    train_demo  = [train_dataset[i] for i in train_indices]
    test_demo = [train_dataset[i] for i in test_indices]
        
    return train_demo,test_demo