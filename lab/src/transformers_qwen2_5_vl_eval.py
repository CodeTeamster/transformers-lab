from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from utils import calculate_inference_time, calculate_flops


import argparse
import os


@calculate_inference_time(verbosity=False)
def inference(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    conversation: list,
):
    # Preparation for inference
    # text: str =
    # <|im_start|>system
    # You are a helpful assistant.<|im_end|>
    # <|im_start|>user
    # <|vision_start|><|image_pad|><|vision_end|>Tell me what I should pay attention to.<|im_end|>
    # <|im_start|>assistant
    text = processor.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=False,
    )

    # image_inputs = [PIL.ImageFile, ...] or None
    image_inputs, video_inputs = process_vision_info(conversation)

    assert isinstance(image_inputs, list) or image_inputs is None
    # inputs = {'input_ids': tensor(), 'attention_mask': tensor(), 'pixel_values': tensor(), 'image_grid_thw': tensor()}
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    return len(generated_ids_trimmed[0]), processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def run_eval(args: argparse.Namespace):
    if args.divprune_subset_ratio > 0.0:
        os.environ['SUBSET_RATIO'] = str(args.divprune_subset_ratio)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
        torch_dtype="auto",
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = Qwen2_5_VLProcessor.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
        use_fast=False,
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": args.image_path,
                },
                {
                    "type": "text",
                    "text": args.text_prompt,
                },
            ],
        }
    ]

    flops, params = calculate_flops(model)
    inference_time, (num_tokens, inference_res) = inference(
        model=model,
        processor=processor,
        conversation=conversation,
    )
    tps = num_tokens / inference_time

    return flops, params, inference_time, tps, inference_res


def defaultargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pretrained-model-path',
        type=str,
        default="/home/yrc/workspace/transformers/experiments/checkpoints/Qwen2.5-VL-7B-Instruct",
        help='Path to the pretrained model',
    )
    parser.add_argument(
        '--divprune-subset-ratio',
        type=float,
        default=0.098,
        help='Subset ratio for divPrune, default is 0.0',
    )
    parser.add_argument(
        '--image-path',
        type=str,
        default="/home/yrc/workspace/transformers/experiments/dataset/imagenet1k/val/n01440764/ILSVRC2012_val_00048969.JPEG",
        # default="/home/yrc/workspace/transformers/experiments/dataset/imagenet1k/val/n01440764/ILSVRC2012_val_00000293.JPEG",
        help='Path to the image for evaluation',
    )
    parser.add_argument(
        '--text-prompt',
        type=str,
        default="Tell me what I should pay attention to.",
        help='Text prompt for the model',
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = defaultargs()
    flops, params, inference_time, tps, inference_res = run_eval(args)

    print(f"Flops: {flops}, Params: {params}")
    print(f"Inference Time: {inference_time: .2f} s")
    print(f"TPS: {tps: .2f} tokens/s")
    print(f"Inference Output: {inference_res}")