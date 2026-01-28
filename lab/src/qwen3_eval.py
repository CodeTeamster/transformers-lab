from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
from utils import calculate_inference_time


import argparse
import os
import random
import json


@calculate_inference_time(verbosity=False)
def inference(
    model: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    conversation: list,
):
    # Preparation for inference
    # text: str =
    # <|im_start|>system
    # You are a helpful assistant.<|im_end|>
    # <|im_start|>user
    # Give me a short introduction to large language model.<|im_end|>
    # <|im_start|>assistant
    # <think>
    #
    # </think>
    text = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # inputs = {'input_ids': tensor(), 'attention_mask': tensor()}
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Inference: Generation of the output
    cont = model.generate(**inputs, max_new_tokens=256)

    if os.environ.get('RANDOM_DISCARD') is not None:
        random_discard = json.loads(os.environ['RANDOM_DISCARD'])
        inputs.input_ids, _, _, _ = model.prepare_inputs_for_discard(
            input_ids=inputs.input_ids,
            discard_rate=random_discard['discard_rate'],
            discard_before_layer=random_discard['discard_before_layer'],
        )
    generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
    generated_texts = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    generated_token_lens = [
        tokenizer(
            generated_text,
            return_tensors="pt",
        ).input_ids.shape[-1] for generated_text in generated_texts
    ]

    return generated_token_lens, generated_texts


def run_eval(args: argparse.Namespace):
    os.environ.pop("RANDOM_DISCARD", None)

    model = Qwen3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
    )

    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": args.text_prompt,
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

    inference_time, (num_tokens, inference_res) = inference(
        model=model,
        tokenizer=tokenizer,
        conversation=conversation,
    )
    tps = num_tokens[0] / inference_time

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
        default="/home/jovyan/nas/yrc/model/Qwen/Qwen3-8B",
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
        '--text-prompt',
        type=str,
        default="Are you a large language model? Just answer yes or no.",
        help='Text prompt for the model',
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = defaultargs()
    run_eval(args)