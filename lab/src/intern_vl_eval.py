from transformers import InternVLForConditionalGeneration, InternVLProcessor, BitsAndBytesConfig
from utils import calculate_inference_time, calculate_flops


import argparse
import os
import random
import json
import torch


@calculate_inference_time(verbosity=False)
def inference(
    model: InternVLForConditionalGeneration,
    processor: InternVLProcessor,
    conversation: list,
):
    # Preparation for inference
    # text: str =
    # <|im_start|>system
    # You are a helpful assistant.<|im_end|>
    # <|im_start|>user
    # <IMG_CONTEXT>
    # Tell me what I should pay attention to.<|im_end|>
    # <|im_start|>assistant
    inputs = processor.apply_chat_template(
        conversation=conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    # image: 1500 * 1435
    # inputs["input_ids"]: (1, 2619)
    # image_id len: (2560)
    # inputs["attention_mask"]: (1, 2619)
    # inputs["pixel_values"]: (10, 3, 448, 448)

    # Inference: Generation of the output
    full_ids = model.generate(**inputs, max_new_tokens=256)
    full_texts = processor.batch_decode(
        full_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    origin_texts = processor.batch_decode(
        inputs.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    generated_texts = [full_text.replace(origin_text, "") for full_text, origin_text in zip(full_texts, origin_texts)]
    generated_token_lens = [processor.tokenizer(generated_text, return_tensors="pt").input_ids.shape[-1] for generated_text in generated_texts]

    return generated_token_lens, generated_texts


def run_eval(args: argparse.Namespace):
    os.environ.pop("RANDOM_DISCARD", None)

    model = InternVLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
        dtype="bfloat16",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )

    processor = InternVLProcessor.from_pretrained(
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
        discard_before_layers = [False] * model.config.text_config.num_hidden_layers
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
    inference_time, (num_tokens, inference_res) = inference(
        model=model,
        processor=processor,
        conversation=conversation,
    )
    tps = num_tokens[0] / inference_time

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
        default="/home/jovyan/nas/yrc/model/OpenGVLab/InternVL3_5-8B-HF",
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
        default=0.1,
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
        default=None,
    )
    parser.add_argument(
        '--image-path',
        type=str,
        default="./resources/ai2d-demo.JPEG",
        # default="./resources/ILSVRC2012_val_00048969.JPEG",
        help='Path to the image for evaluation',
    )
    parser.add_argument(
        '--text-prompt',
        type=str,
        default="What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud, only give the name corresponding to the correct option.",
        # default="Please describe the detail of this image.",
        help='Text prompt for the model',
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = defaultargs()
    run_eval(args)