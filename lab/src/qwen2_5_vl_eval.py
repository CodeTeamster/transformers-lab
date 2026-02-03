from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from utils import calculate_inference_time, calculate_flops, visualize_attention


import argparse
import os
import random
import json
import gc
import numpy as np


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
    image_inputs, _ = process_vision_info(conversation)

    assert isinstance(image_inputs, list) or image_inputs is None
    # inputs = {'input_ids': tensor(), 'attention_mask': tensor(), 'pixel_values': tensor(), 'image_grid_thw': tensor()}
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    full_ids = model.generate(
        **inputs,
        return_dict_in_generate=True,
        output_attentions=True,
        max_new_tokens=256,
    )
    full_texts = processor.batch_decode(
        full_ids.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    origin_texts = processor.batch_decode(
        inputs.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    generated_texts = [full_text.replace(origin_text, "") for full_text, origin_text in zip(full_texts, origin_texts)]
    generated_token_lens = [processor.tokenizer(generated_text, return_tensors="pt").input_ids.shape[-1] for generated_text in generated_texts]

    return generated_token_lens, generated_texts, full_ids.attentions


def run_eval(args: argparse.Namespace):
    os.environ.pop("RANDOM_DISCARD", None)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
        dtype="bfloat16",
        attn_implementation="eager",
        device_map="cuda:0",
    )

    processor = Qwen2_5_VLProcessor.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
        use_fast=True,
    )

    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                },
            ],
        },
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

    if abs(args.discard_rate) > 0.0 and len(args.discard_before_layers) > 0:
        discard_before_layers = [False] * model.config.num_hidden_layers
        for layer_id in args.discard_before_layers:
            discard_before_layers[layer_id] = True
        if args.discard_rate < 0.0:
            discard_rate = random.uniform(0.0, abs(args.discard_rate))
        else:
            discard_rate = args.discard_rate
        random_discard = {
            "discard_rate": discard_rate,
            "discard_before_layer": discard_before_layers,
            "discard_seed": args.discard_seed,
        }
        os.environ['RANDOM_DISCARD'] = json.dumps(random_discard)

    flops, params = calculate_flops(model)
    inference_time, (num_tokens, inference_res, multihead_attentions) = inference(
        model=model,
        processor=processor,
        conversation=conversation,
    )
    tps = num_tokens[0] / inference_time

    if args.visualization_output_dir is not None:
        os.makedirs(args.visualization_output_dir, exist_ok=True)
        for layer in range(0, model.config.text_config.num_hidden_layers):
            output_path = os.path.join(
                args.visualization_output_dir,
                f"attn_heatmap_layer-{layer + 1}_discard-{args.discard_rate}.png",
            )
            visualize_attention(
                multihead_attention = multihead_attentions[0][layer].cpu(),
                output_path=output_path,
                title=f"Layer {layer + 1}",
                pool_kernel_size=10,
            )

    print(f'Model FLOPs: {flops} ')
    print(f'Model Params: {params} ')
    print(f'Inference time: {inference_time:.2f} s')
    print(f'Token generated per second: {tps:.2f} token/s')
    print(f'Inference result: {inference_res[0]}')

    os.environ.pop("RANDOM_DISCARD", None)

    return


def defaultargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pretrained-model-path',
        type=str,
        default="/home/jovyan/nas/yrc/model/Qwen/Qwen2.5-VL-7B-Instruct",
        help='Path to the pretrained model',
    )
    parser.add_argument(
        '--divprune-subset-ratio',
        type=float,
        default=0.098,
        help='Subset ratio for divPrune, default is 0.0',
    )
    parser.add_argument(
        '--discard-rate',
        type=float,
        default=0.,
        help='Token discarding rate',
    )
    parser.add_argument(
        '--discard-before-layers',
        type=int,
        nargs='+',
        default=[0],
    )
    parser.add_argument(
        '--discard-seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--image-path',
        type=str,
        default="./resources/elephant_512x512.JPEG",
        # default="./resources/ILSVRC2012_val_00048969.JPEG",
        help='Path to the image for evaluation',
    )
    parser.add_argument(
        '--text-prompt',
        type=str,
        default="How many elephants can you see in this image, just answer a number?",
        # default="Please describe the detail of this image.",
        help='Text prompt for the model',
    )
    parser.add_argument(
        '--visualization-output-dir',
        type=str,
        default="./workdir/attention_visualization/qwen2.5-vl-7b",
        # default=None,
        help='Directory to save visualization outputs',
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = defaultargs()

    for rate in np.arange(0.0, 1.0, 0.1):
        args.discard_rate = round(rate, 1)
        print(f'================ Discard Rate: {args.discard_rate} ================')
        run_eval(args)
        gc.collect()