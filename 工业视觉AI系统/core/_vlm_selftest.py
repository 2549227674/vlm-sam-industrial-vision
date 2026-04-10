"""core.vlm 的极简自检脚本：不依赖网络/Key，验证解析逻辑不会崩。"""

from __future__ import annotations

from core.vlm import _parse_vlm_output


def main():
    text = (
        "TAGS_EN: bent lead, missing part, transistor, metal screw\n"
        "DESC_EN: A close-up of a small electronic component; one pin looks bent.\n"
        "DESC_ZH: 这是一张电子元件的近景图，其中一根引脚可能存在弯折。\n"
    )

    out = _parse_vlm_output(text, max_tags=6)
    assert out.tags_en[:2] == ["bent lead", "missing part"]
    assert "electronic" in out.desc_en.lower()
    assert "引脚" in out.desc_zh


if __name__ == "__main__":
    main()

