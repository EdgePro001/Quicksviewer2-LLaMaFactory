# processing_glm4v_withCube.py
"""
Processor class for GLM4V with Cubing support.
"""

from typing import Any, List, Optional, Union

import torch
from transformers.image_utils import ImageInput
from transformers.video_utils import VideoInput

from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType

from .configuration_glm4v_withCube import Glm4vConfig

from transformers import AutoProcessor


class Glm4vProcessor(ProcessorMixin):
    r"""
    Constructs a GLM4V processor which wraps a GLM4V image processor and a GLM4V tokenizer into a single processor.

    [`Glm4vProcessor`] offers all the functionalities of [`Glm4vCubingImageProcessor`] and [`PreTrainedTokenizer`].
    See the [`~Glm4vProcessor.__call__`] and [`~Glm4vProcessor.decode`] for more information.

    Args:
        image_processor ([`Glm4vCubingImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer is a required input.
        config ([`Glm4vConfig`], *optional*):
            The model config for accessing Cubing parameters.
        auto_videos_bound (`bool`, *optional*, defaults to `True`):
            Whether to automatically construct videos_bound from video_grid_thw.
            Set to False if you want to manually provide videos_bound.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Glm4vCubingImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self, 
        image_processor, 
        tokenizer, 
        config: Optional[Glm4vConfig] = None,
        auto_videos_bound: bool = True,
        **kwargs
    ):
        # 不调用 super().__init__，避免类型检查
        # super().__init__(image_processor, tokenizer)  
        
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        
        # 设置自定义属性
        self.config = config
        self.auto_videos_bound = auto_videos_bound
        
        # 设置 _in_target_context_manager（ProcessorMixin 需要）
        self._in_target_context_manager = False

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        return_videos_bound: bool = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        **kwargs,
    ):
        """
        Main method to prepare for the model one or several sequences(s) and image(s)/video(s).

        Args:
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string).
            images (`PIL.Image.Image`, `List[PIL.Image.Image]`):
                The image or batch of images to be prepared. Each image should be a PIL image.
            videos (`List[PIL.Image.Image]`, `List[List[PIL.Image.Image]]`):
                The video or batch of videos to be prepared. Each video should be a list of PIL images (frames).
            return_videos_bound (`bool`, *optional*):
                Whether to return videos_bound in the output.
                If None, defaults to self.auto_videos_bound.
                Set to False to let the model auto-infer from video_grid_thw.
            padding (`bool`, `str`, *optional*, defaults to `False`):
                Padding strategy for tokenizer.
            truncation (`bool`, `str`, *optional*):
                Truncation strategy for tokenizer.
            max_length (`int`, *optional*):
                Maximum length for tokenizer.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                Return tensor type.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to.
            - **pixel_values** -- Pixel values of images to be fed to a model.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model.
            - **image_grid_thw** -- Grid shape (temporal, height, width) for images.
            - **video_grid_thw** -- Grid shape (temporal, height, width) for videos.
            - **videos_bound** -- (Optional) Frame boundaries for videos.
        """
        # ========== Step 1: Process images (unchanged from native GLM4V) ==========
        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors=return_tensors)

        # ========== Step 2: Process videos (pixel-level processing unchanged) ==========
        video_inputs = {}
        videos_bound = None
        
        if videos is not None:
            # Process video pixels (same as native GLM4V)
            video_inputs = self.image_processor(images=videos, return_tensors=return_tensors)
            if 'image_grid_thw' in video_inputs:
                video_inputs['video_grid_thw'] = video_inputs.pop('image_grid_thw')
    
            if 'pixel_values' in video_inputs:
                video_inputs['pixel_values_videos'] = video_inputs.pop('pixel_values')
            
            # ✨ Calculate token counts based on mode
            if self.config is not None and self.config.use_cubing:
                # Cubing mode: calculate dynamic token count
                num_video_tokens, videos_bound = self._calculate_video_tokens_cubing(
                    video_inputs['video_grid_thw']
                )
            else:
                # Native mode: use original formula
                num_video_tokens = self._calculate_video_tokens_native(
                    video_inputs['video_grid_thw']
                )
                
                # Optionally construct videos_bound for auto-inference
                if self.auto_videos_bound:
                    _, videos_bound = self._construct_videos_bound(
                        video_inputs['video_grid_thw']
                    )
            
            # Store for later use in text processing
            video_inputs['_num_placeholder_tokens'] = num_video_tokens

        # ========== Step 3: Process text with correct placeholder counts ==========
        if text is not None:
            # Get placeholder token counts
            num_image_tokens = self._get_num_image_tokens(image_inputs)
            num_video_tokens = video_inputs.get('_num_placeholder_tokens', 0)
            
            # Replace <image>/<video> tags with correct number of image tokens
            if isinstance(text, str):
                text = [text]
            
            processed_text = []
            for t in text:
                # This is a simplified placeholder - actual implementation 
                # would need to handle the specific token format of GLM4V
                t = self._insert_image_tokens(t, num_image_tokens, num_video_tokens)
                processed_text.append(t)
            
            # Tokenize
            text_inputs = self.tokenizer(
                processed_text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                **kwargs,
            )
        else:
            text_inputs = {}

        # ========== Step 4: Combine all inputs ==========
        encoding = {**text_inputs, **image_inputs, **video_inputs}
        
        # Decide whether to return videos_bound
        if return_videos_bound is None:
            return_videos_bound = self.auto_videos_bound
        
        if videos_bound is not None and return_videos_bound:
            encoding['videos_bound'] = videos_bound
        
        # Clean up internal keys
        encoding.pop('_num_placeholder_tokens', None)
        
        return encoding

    def _calculate_video_tokens_native(self, video_grid_thw: torch.Tensor) -> int:
        """
        Calculate placeholder tokens for native GLM4V mode.
        
        Args:
            video_grid_thw: [num_videos, 3] - (temporal, height, width) for each video
        
        Returns:
            Total number of placeholder tokens
        """
        if self.config is None:
            # Fallback: use default spatial_merge_size=2
            spatial_merge_size = 2
        else:
            spatial_merge_size = self.config.vision_config.spatial_merge_size
        
        total_tokens = 0
        for t, h, w in video_grid_thw:
            # Original GLM4V formula
            tokens_per_video = (t.item() * h.item() * w.item()) // (spatial_merge_size ** 2)
            total_tokens += tokens_per_video
        
        return total_tokens

    def _calculate_video_tokens_cubing(self, video_grid_thw: torch.Tensor) -> tuple[int, list]:
        """
        Calculate placeholder tokens for Cubing mode.
        
        Args:
            video_grid_thw: [num_videos, 3] - (temporal, height, width) for each video
        
        Returns:
            tuple:
                - total_tokens: Total number of placeholder tokens
                - videos_bound: List of (start_frame, end_frame) tuples
        """
        if self.config is None:
            raise ValueError("Config is required for Cubing mode")
        
        total_tokens = 0
        videos_bound = []
        current_frame_idx = 0
        
        for t, h, w in video_grid_thw:
            num_frames = t.item()
            
            # Calculate number of cubes for this video
            # Formula from Cubing module: max(round(N_frames / fpq) - 1, 1)
            num_cubes = max(round(num_frames / self.config.cubing_fpq) - 1, 1)
            
            # Tokens per video = num_cubes * queries_per_cube
            tokens_per_video = num_cubes * self.config.resampler_num_queries
            
            # Add thumbnail token if enabled
            if self.config.cubing_use_thumbnail:
                tokens_per_video += 1
            
            total_tokens += tokens_per_video
            
            # Record frame boundaries for this video
            videos_bound.append((current_frame_idx, current_frame_idx + num_frames))
            current_frame_idx += num_frames
        
        return total_tokens, videos_bound

    def _construct_videos_bound(self, video_grid_thw: torch.Tensor) -> tuple[int, list]:
        """
        Construct videos_bound from video_grid_thw
        
        This is used when videos_bound is not provided but auto-inference is desired.
        Assumes each entry in video_grid_thw represents a separate video.
        
        Args:
            video_grid_thw: [num_videos, 3] - (temporal, height, width) for each video
        
        Returns:
            tuple:
                - num_videos: int - Number of videos
                - videos_bound: list - List of (start_frame, end_frame) tuples
        
        Example:
            >>> video_grid_thw = torch.tensor([[120, 24, 24], [80, 24, 24]])
            >>> num_videos, videos_bound = self._construct_videos_bound(video_grid_thw)
            >>> print(videos_bound)
            [(0, 120), (120, 200)]
        """
        videos_bound = []
        current_frame_idx = 0
        
        for t, h, w in video_grid_thw:
            num_frames = t.item()
            videos_bound.append((current_frame_idx, current_frame_idx + num_frames))
            current_frame_idx += num_frames
        
        return len(videos_bound), videos_bound

    def _get_num_image_tokens(self, image_inputs: dict) -> int:
        """
        Calculate number of placeholder tokens for images (unchanged from native).
        
        Args:
            image_inputs: Dictionary containing image_grid_thw
        
        Returns:
            Total number of image placeholder tokens
        """
        if not image_inputs or 'image_grid_thw' not in image_inputs:
            return 0
        
        image_grid_thw = image_inputs['image_grid_thw']
        
        if self.config is None:
            spatial_merge_size = 2
        else:
            spatial_merge_size = self.config.vision_config.spatial_merge_size
        
        total_tokens = 0
        for t, h, w in image_grid_thw:
            # Same formula as native GLM4V
            tokens_per_image = (t.item() * h.item() * w.item()) // (spatial_merge_size ** 2)
            total_tokens += tokens_per_image
        
        return total_tokens

    def _insert_image_tokens(
        self, 
        text: str, 
        num_image_tokens: int, 
        num_video_tokens: int
    ) -> str:
        """
        Replace <image> and <video> placeholders with actual image token IDs.
        
        NOTE: This is a simplified implementation. The actual GLM4V processor
        has more complex logic for handling special tokens and formatting.
        
        Args:
            text: Input text with <image> and <video> tags
            num_image_tokens: Number of tokens to insert for each <image>
            num_video_tokens: Number of tokens to insert for each <video>
        
        Returns:
            Text with placeholders replaced
        """
        # Get image token from tokenizer
        if hasattr(self.tokenizer, 'image_token'):
            image_token = self.tokenizer.image_token
        else:
            # Fallback to GLM4V's default
            image_token = "<image>"
        
        # Replace <video> with N image tokens
        if '<video>' in text and num_video_tokens > 0:
            video_placeholder = image_token * num_video_tokens
            text = text.replace('<video>', video_placeholder)
        
        # Replace <image> with N image tokens (if needed)
        # Note: In GLM4V, <image> might already be the token itself
        # This part depends on the specific tokenizer implementation
        
        return text

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        # videos_bound is optional, so it's included in the list
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + ["videos_bound"]))
    
__all__ = ["Glm4vProcessor"]
AutoProcessor.register(Glm4vConfig, Glm4vProcessor, exist_ok=True)