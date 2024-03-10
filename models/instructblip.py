from typing import List

from PIL import Image
import torch

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

#%%
from open_flamingo.eval.eval_model import BaseEvalModel
from torchvision import transforms
from transformers.image_utils import (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
class EvalModel(BaseEvalModel):
    """BLIP-2 model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "cpu"
    """

    def __init__(self, model_args):
        assert (
            "processor_path" in model_args
            and "lm_path" in model_args
        
            and "device" in model_args
        ), "BLIP-2 requires processor_path, lm_path, and device arguments to be specified"

        model_args["device"] = int(model_args["device"])

        self.device = model_args["device"] if model_args["device"] >= 0 else "cpu"
        self.processor = InstructBlipProcessor.from_pretrained(model_args["processor_path"])
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_args["lm_path"]
        ).half()
        self.tokenizer = self.processor.tokenizer
        self.qformer_tokenizer = self.processor.qformer_tokenizer
        self.model.to(self.device)
        self.model.eval()
        self.processor.tokenizer.padding_side = "left"

    def _prepare_images(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, channels, height, width).
        """
        batch_images = None
        
        # assert all(
        #     len(example) == 1 for example in batch
        # ), "BLIP-2 only supports one image per example"

        for example in batch:
            # assert len(example) == 1, "BLIP-2 only supports one image per example"
            batch_images = torch.cat(
                [
                    batch_images,
                    self.processor.image_processor(example, return_tensors="pt")[
                        "pixel_values"
                    ],
                ]
                if batch_images is not None
                else [
                    self.processor.image_processor(example, return_tensors="pt")[
                        "pixel_values"
                    ]
                ],
                dim=0,
            )
        return batch_images
    def _prepare_images_no_normalize(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, channels, height, width).
        """
        batch_images = None
        # assert all(
        #     len(example) == 1 for example in batch
        # ), "BLIP-2 only supports one image per example"

        for example in batch:
            # assert len(example) == 1, "BLIP-2 only supports one image per example"
            batch_images = torch.cat(
                [
                    batch_images,
                    self.processor.image_processor(example, do_normalize = False,return_tensors = "pt")[
                        "pixel_values"
                    ],
                ]
                if batch_images is not None
                else [
                    self.processor.image_processor(example, do_normalize = False,return_tensors = "pt")[
                        "pixel_values"
                    ],
                
                ],
                dim=0,
            )
        # print(torch.max(batch_images),torch.min(batch_images))
        return batch_images
    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        print("batch_images is",batch_images)
        # if type(batch_images) is list:
            # batch_images = batch_images[0]
        inputs= self.processor(
           images=batch_images, text=batch_text, truncation=True,
           padding = "longest",return_tensors="pt").to(self.device)   
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_length=10, 
                # truncation=True,
                # padding = True,
                num_beams=num_beams,
                length_penalty=length_penalty               
            )

        return self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_outputs_attack(
        self,
        attack: torch.Tensor,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        # while type(batch_images) is list:
            # batch_images = batch_images[0]
        
        print("attacked batch_images is",batch_images)

        
        
        inputs= self.processor(
           images=batch_images, text=batch_text,padding = "longest", truncation=True, return_tensors="pt").to(self.device)   
        inputs.pop('pixel_values')
        inputs['pixel_values'] = self._prepare_images(batch_images).to(self.device)
        
        attack = attack.to(self.device)
        # print("attack shape is",attack.shape)
        input_x = self._prepare_images_no_normalize(batch_images).to(self.device)
        input_x += attack        
        normalizer = transforms.Normalize(mean= OPENAI_CLIP_MEAN,std = OPENAI_CLIP_STD)
        input_x = normalizer(input_x)
        inputs['pixel_values'] = input_x
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_length=10,
                
                num_beams=num_beams,
                length_penalty=length_penalty
            )

        return self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    
    def get_vqa_prompt(self, question, answer=None) -> str:
        return (
            f"Question:{question} Answer:{answer if answer is not None else ''}"
        )

    def get_caption_prompt(self, caption=None) -> str:
        return ""#f"Output:{caption if caption is not None else ''}{'</s>' if caption is not None else ''}"

    def get_classification_prompt(self, class_str=None) -> str:
        return ""#f"A photo of a {class_str if class_str is not None else ''}{'</s>' if class_str is not None else ''}"

