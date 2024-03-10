from typing import List

from PIL import Image
import torch
from torchvision import transforms
from utils.eval_model import BaseEvalModel
from models.flamingo_src.factory import create_model_and_transforms

class EvalModel(BaseEvalModel):
    """OpenFlamingo model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "CPU"
    """

    def __init__(self, model_args):
        assert (
            "vision_encoder_path" in model_args
            and "lm_path" in model_args
            and "device" in model_args
            and "checkpoint_path" in model_args
            and "lm_tokenizer_path" in model_args
            and "cross_attn_every_n_layers" in model_args
            and "vision_encoder_pretrained" in model_args
        ), "OpenFlamingo requires vision_encoder_path, lm_path, device, checkpoint_path, lm_tokenizer_path, cross_attn_every_n_layers, and vision_encoder_pretrained arguments to be specified"

        model_args["device"] = int(model_args["device"])
        self.device = model_args["device"] if model_args["device"] >= 0 else "cpu"
        (
            self.model,
            self.image_processor,
            self.tokenizer,
        ) = create_model_and_transforms(
            model_args["vision_encoder_path"],
            model_args["vision_encoder_pretrained"],
            model_args["lm_path"],
            model_args["lm_tokenizer_path"],
            cross_attn_every_n_layers=int(model_args["cross_attn_every_n_layers"]),
        )
        checkpoint = torch.load(model_args["checkpoint_path"], map_location="cpu")
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.padding_side = "left"

    def _prepare_images(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = self.image_processor(image)

                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        return batch_images
    def _aug_images(self, batch: List[List[torch.Tensor]],aug = None) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.
            aug: the function for data augmentation
        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        print("the shape for the image to be augmented is",batch[0][-1].shape)
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                # only aug the last image, as it is the image to be attacked
                assert image.shape[0]==1, "expect the frame dim to be 1"
                assert len(image[0].shape)==3,"expect to get a 3d image"
                
                if iimage == len(example)-1:
                    preprocessed = aug(image[0])
                else:
                    preprocessed = image[0]
                    
                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        return batch_images
    def _prepare_images_no_normalize(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess images and stack them.    

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                if len(self.image_processor.transforms)==5:
                    self.image_processor.transforms = self.image_processor.transforms[:-1]
                assert transforms.Normalize not in [type(x) for x in self.image_processor.transforms] and len(self.image_processor.transforms)==4
                
                preprocessed = self.image_processor(image)
                assert torch.min(preprocessed)>=0 and torch.max(preprocessed)<=1
                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        return batch_images
    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
        top_p: float = 1.0,
        top_k: int = 1,
        do_sample: bool = False
    ) -> List[str]:
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.inference_mode():
            outputs = self.model.generate(
                self._prepare_images_no_normalize(batch_images).to(self.device),
                input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                top_p=top_p,
                top_k=top_k,
                do_sample = do_sample
            )
        # print("raw outputs before slicing",outputs)
        outputs = outputs[:, len(input_ids[0]) :]
        """
        print("raw outputs",outputs)
        print("decoded",self.tokenizer.batch_decode(outputs, skip_special_tokens=True))
        print("======end=======")
        """
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    
    def get_outputs_attack(
        self,
        attack: torch.Tensor,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
         top_p: float = 1.0,
         top_k: int = 1,
         do_sample = False,
         augmentation = None,
    ) -> List[str]:
        
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        attack = attack.to(self.device)
        input_x = self._prepare_images_no_normalize(batch_images).to(self.device)
        # print("input_x",input_x.shape)
        # print("attack",attack.shape)
        # input_x[0,-1] = input_x[0,-1] + attack
        input_x[:, -1, :, :, :, :] += attack        
        if augmentation is not None:
            input_x = self._aug_images(input_x,augmentation)
            input_x = input_x.to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_x,
                input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                top_p=top_p,
                top_k=top_k,
                do_sample = do_sample,
                
            )
        
        outputs = outputs[:, len(input_ids[0]) :]
        
        # print("raw outputs",outputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # return outputs
    
    def get_vqa_prompt(self, question, answer=None) -> str:
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_caption_prompt(self, caption=None) -> str:
        return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

    def get_classification_prompt(self, class_str=None) -> str:
        return f"<image>A photo of a {class_str if class_str is not None else ''}{'<|endofchunk|>' if class_str is not None else ''}"
