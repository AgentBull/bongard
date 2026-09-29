"""Native T5Gemma2 image preprocessing and trusted visual-token insertion."""

import base64
import binascii
import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import Gemma3Processor

from .data import DataError


def decode_images(sources):
    images, contents = [], []
    for index, source in enumerate(sources):
        try:
            header, encoded = source.split(",", 1)
            content = base64.b64decode(encoded, validate=True)
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as image:
                    if Image.MIME.get(image.format) != header[5:-7]:
                        raise ValueError("Image content does not match the declared media type")
                    if getattr(image, "n_frames", 1) != 1:
                        raise ValueError("Expected a still image, not an animation")
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
            contents.append(content)
        except (
            ValueError,
            OSError,
            binascii.Error,
            UnidentifiedImageError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise DataError(f"images[{index}]: invalid image: {exc}") from exc
    return images, tuple(contents)


def image_options(image_processor):
    return {
        name: getattr(image_processor, name)
        for name in (
            "do_pan_and_scan", "pan_and_scan_min_crop_size",
            "pan_and_scan_max_num_crops", "pan_and_scan_min_ratio_to_activate",
        )
        if getattr(image_processor, name) is not None
    }


def image_pixels(sources, image_processor):
    """Prepare pixels for stored token IDs without invoking any tokenizer."""
    images, contents = decode_images(sources)
    encoded = image_processor(images=[images], return_tensors="pt", **image_options(image_processor))
    return encoded.pixel_values, contents


def compile_images(sources, tokenizer, config, image_processor):
    """Keep encoded image content in the state identity, never in text tokens."""
    expected = (config.boi_token_index, config.image_token_index, config.eoi_token_index)
    actual = tuple(
        getattr(tokenizer, name, None)
        for name in ("boi_token_id", "image_token_id", "eoi_token_id")
    )
    if actual != expected or len(set(expected)) != 3:
        raise DataError("Tokenizer image controls do not match the native vision configuration")
    images, contents = decode_images(sources)
    processor = Gemma3Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_seq_length=config.mm_tokens_per_image,
    )
    # All text here is compiler-owned. Arbitrary state/question text still goes
    # through GuardedTokenizer, so a literal image marker cannot consume pixels.
    prompt = "\n".join(f"image {i + 1}: {tokenizer.boi_token}" for i in range(len(images)))
    encoded = processor(
        text=[prompt],
        images=[images],
        add_special_tokens=False,
        return_tensors="pt",
        images_kwargs=image_options(image_processor),
    )
    return tuple(encoded.input_ids[0].tolist()), encoded.pixel_values, tuple(contents)
