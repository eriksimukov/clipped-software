"""Zeichnet die Endkarte mit den Kanaelen auf dem unscharfen letzten Bild des Clips."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from clipped.config import ASSETS_DIR, Channel

TEXT_FONT = ASSETS_DIR / "fonts" / "Anton-Regular.ttf"
ICON_FONT = ASSETS_DIR / "fonts" / "fa-brands-400.ttf"
# Font Awesome Brands; Plattformen ohne Eintrag bekommen nur ihren Namen als Text.
ICONS = {
    "tiktok": "",
    "youtube": "",
    "instagram": "",
    "twitch": "",
    "x": "",
    "twitter": "",
    "facebook": "",
    "discord": "",
}
TITLE_Y = 640
FIRST_ROW_Y = 830
ROW_HEIGHT = 135


def compose_endcard(
    background: Path | None, title: str, channels: list[Channel], out_path: Path, size=(1080, 1920)
) -> None:
    width, _height = size
    if background and background.exists():
        with Image.open(background) as source:
            image = ImageOps.fit(source.convert("RGB"), size)
        image = image.filter(ImageFilter.GaussianBlur(32))
        image = ImageEnhance.Brightness(image).enhance(0.38)
    else:
        image = Image.new("RGB", size, (14, 14, 18))

    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(str(TEXT_FONT), 118)
    handle_font = ImageFont.truetype(str(TEXT_FONT), 76)
    icon_font = ImageFont.truetype(str(ICON_FONT), 72)
    draw.text(
        (width / 2, TITLE_Y),
        title.upper(),
        font=title_font,
        fill="white",
        anchor="mm",
        stroke_width=6,
        stroke_fill="black",
    )

    y = FIRST_ROW_Y
    for channel in channels:
        glyph = ICONS.get(channel.platform.lower())
        label = channel.handle if glyph else f"{channel.platform}: {channel.handle}"
        text_width = draw.textlength(label, font=handle_font)
        icon_width = draw.textlength(glyph, font=icon_font) + 30 if glyph else 0
        x = (width - text_width - icon_width) / 2
        if glyph:
            draw.text((x, y), glyph, font=icon_font, fill="white", anchor="lm")
        draw.text(
            (x + icon_width, y),
            label,
            font=handle_font,
            fill="white",
            anchor="lm",
            stroke_width=3,
            stroke_fill="black",
        )
        y += ROW_HEIGHT

    image.save(out_path)
