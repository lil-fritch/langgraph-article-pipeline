from io import BytesIO
from pathlib import Path

import aiohttp
from PIL import Image

from src.config import OUTPUT_DIR, settings
from src.llm_client import LLMClient
from src.utils.logger import setup_logger

logger = setup_logger("image_generator")


def build_image_prompt(topic: str, topic_specs: dict | None, outline: dict | None) -> str:
    subject_name = None
    if topic_specs:
        subject_name = topic_specs.get("name") if isinstance(topic_specs, dict) else getattr(topic_specs, "name", None)

    title = (outline or {}).get("main_title") or topic
    base_subject = title
    if subject_name and subject_name.lower() not in base_subject.lower():
        base_subject = f"{base_subject} featuring {subject_name}"

    style = (
        "Minimal editorial style, clean geometric shapes, limited palette, soft gradients, "
        "modern composition, high contrast focal point, cinematic lighting, widescreen 16:9 cover, "
        "magazine cover layout, clear top margin reserved for headline, subtle framing, vignette, "
        "strong central subject, unique visual concept tied to the topic, "
        "no text, no logos, no watermarks, no UI, no brand names."
    )
    return f"{base_subject}. {style}"


async def generate_article_cover(
    llm: LLMClient,
    topic: str,
    topic_specs: dict | None,
    outline: dict | None,
    safe_name: str,
    save_to_disk: bool = False,
) -> str:
    """Generate a 16:9 cover image for the article via the OpenAI Images API."""
    image_path = None
    if save_to_disk:
        images_dir = Path(OUTPUT_DIR) / "images" / "covers"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_path = images_dir / f"{safe_name}.webp"
        if image_path.exists():
            return str(image_path)

    prompt = build_image_prompt(topic, topic_specs, outline)
    logger.info(f"Cover prompt: {prompt}")

    image_url = await llm.async_generate_image(prompt)
    if not image_url:
        logger.warning("Cover generation failed: no image URL returned.")
        return ""

    logger.info(f"Image generated at: {image_url}")
    if not save_to_disk:
        return image_url

    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    logger.warning(f"Cover download failed ({response.status}): {image_url}")
                    return image_url
                image_data = BytesIO()
                async for chunk in response.content.iter_chunked(8192):
                    image_data.write(chunk)
                image_data.seek(0)

        img = Image.open(image_data)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        target_ratio = 16 / 9
        width, height = img.size
        current_ratio = width / height if height else target_ratio
        if current_ratio > target_ratio:
            new_width = int(height * target_ratio)
            left = (width - new_width) // 2
            img = img.crop((left, 0, left + new_width, height))
        elif current_ratio < target_ratio:
            new_height = int(width / target_ratio)
            top = (height - new_height) // 2
            img = img.crop((0, top, width, top + new_height))

        target_width = 1024
        img = img.resize((target_width, int(target_width / target_ratio)), Image.Resampling.LANCZOS)
        img.save(image_path, "WEBP", quality=85)
        return str(image_path)
    except Exception as e:
        logger.warning(f"Cover processing/saving failed: {e}")
        return image_url
