#!/usr/bin/env python3
"""
Simple script to detect whether any person in an image wears glasses by
prompting the Qwen2.5-VL-7B-Instruct model. The script prints `True` if
glasses are detected and `False` otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from ovi.utils.qwen_vl_utils import process_vision_info
except ImportError:
    from wan.utils.qwen_vl_utils import process_vision_info


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
SYSTEM_PROMPT = (
    "You are a careful vision-language assistant. Only answer using the word "
    "'True' or 'False' without additional text."
)
QUESTION_PROMPT = (
    "请判断这张图片中是否有人佩戴任何类型的眼镜（包括近视镜、太阳镜、防护镜等）。"
    "如果存在至少一位人物佩戴眼镜，请只回答 True；如果没有任何人物佩戴眼镜，请只回答 False。"
)


def resolve_device(device_str: Optional[str]) -> torch.device:
    """Pick the desired torch.device."""
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_dtype(device: torch.device) -> torch.dtype:
    """Select an appropriate dtype for inference on the given device."""
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device.index or 0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def build_messages(image_path: str) -> list[dict]:
    """Construct the multi-modal chat messages expected by the processor."""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": QUESTION_PROMPT,
                },
            ],
        },
    ]


def parse_boolean_answer(answer: str) -> bool:
    """Convert model output to bool, raising on ambiguous responses."""
    normalized = answer.strip().lower()
    if normalized.startswith("true"):
        return True
    if normalized.startswith("false"):
        return False
    raise ValueError(
        f"无法解析模型输出，期望 True/False，实际返回: {answer!r}")


@torch.inference_mode()
def detect_glasses(
    image_path: str,
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    max_new_tokens: int = 32,
) -> tuple[bool, str]:
    """
    Run Qwen-VL to determine whether any person in the image wears glasses.

    Returns:
        (bool, str): Parsed boolean answer and the raw text output.
    """
    resolved_path = Path(image_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"找不到图片文件: {resolved_path}")

    torch_device = resolve_device(device)
    torch_dtype = infer_dtype(torch_device)

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        use_fast=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to(torch_device)

    messages = build_messages(str(resolved_path))
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    model_inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {
        k: (v.to(torch_device) if hasattr(v, "to") else v)
        for k, v in model_inputs.items()
    }

    generated = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )
    trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs["input_ids"], generated)
    ]
    answer = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return parse_boolean_answer(answer), answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use Qwen2.5-VL-7B to detect whether a person wears glasses."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file to inspect.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model name or local path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. cuda:0 or cpu. Defaults to auto selection.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate for the answer.",
    )
    args = parser.parse_args()

    has_glasses, raw_answer = detect_glasses(
        image_path=args.image,
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    print(has_glasses)
    print(f"[raw answer] {raw_answer}")


if __name__ == "__main__":
    main()

