import os
import re
from bs4 import BeautifulSoup, Tag

HTML_FILE = os.path.join(
    os.path.dirname(__file__),
    "财税[2011]137号 财政部 国家税务总局关于免征蔬菜流通环节增值税有关问题的通知[延续执行]"
    "_税 屋——第一时间传递财税政策法规！.html",
)
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "output.md")


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_links(element) -> str:
    """提取元素文本，保留超链接为 Markdown 格式"""
    parts = []
    for child in element.children:
        if isinstance(child, str):
            parts.append(child)
        elif isinstance(child, Tag):
            if child.name == "a":
                href = child.get("href", "")
                link_text = child.get_text()
                if href:
                    parts.append(f"[{link_text}]({href})")
                else:
                    parts.append(link_text)
            else:
                parts.append(extract_links(child))
    result = "".join(parts)
    result = result.replace("\xa0", " ")
    result = re.sub(r"[ \t]+", " ", result)
    return result.strip()


def _expand_table_grid(table_tag) -> list[list[str]]:
    """将 HTML 表格展开为二维文本 grid，正确处理 rowspan/colspan"""
    rows = table_tag.find_all("tr")
    if not rows:
        return []

    max_cols = 0
    for row in rows:
        cols_in_row = 0
        for cell in row.find_all(["td", "th"]):
            cols_in_row += int(cell.get("colspan", 1))
        max_cols = max(max_cols, cols_in_row)

    grid: list[list[str]] = []
    fill: dict[tuple[int, int], str] = {}

    for row_idx, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        grid_row = [""] * max_cols
        cell_idx = 0
        col_pos = 0

        while col_pos < max_cols:
            if (row_idx, col_pos) in fill:
                grid_row[col_pos] = fill[(row_idx, col_pos)]
                col_pos += 1
                continue

            if cell_idx >= len(cells):
                col_pos += 1
                continue

            cell = cells[cell_idx]
            text = clean_text(cell.get_text())
            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))

            for c in range(cs):
                grid_row[col_pos + c] = text
            for r in range(1, rs):
                for c in range(cs):
                    fill[(row_idx + r, col_pos + c)] = text

            cell_idx += 1
            col_pos += cs

        grid.append(grid_row)

    return grid


def html_table_to_narrative(table_tag) -> str:
    """
    将 HTML 表格转为口语化编号列表（复用 extractor/parser.py 的风格），
    同时正确处理 rowspan/colspan。
    以表头为 key，逐行以"字段名：值"方式串联。
    """
    grid = _expand_table_grid(table_tag)
    if not grid:
        return ""

    headers = [re.sub(r"\s+", "", h) for h in grid[0]]
    seen_headers: list[str] = []
    seen_indices: list[int] = []
    for i, h in enumerate(headers):
        if h and h not in seen_headers:
            seen_headers.append(h)
            seen_indices.append(i)

    if not seen_headers:
        return ""

    if len(grid) <= 1:
        return "表头：" + "、".join(seen_headers)

    lines: list[str] = []
    for row_idx, row in enumerate(grid[1:], start=1):
        parts: list[str] = []
        for i, col_idx in enumerate(seen_indices):
            header = seen_headers[i]
            value = row[col_idx] if col_idx < len(row) else ""
            if value:
                parts.append(f"{header}：{value}")
        if parts:
            lines.append(f"（{row_idx}）{'；'.join(parts)}。")

    return "\n".join(lines)


def extract_article_meta(soup: BeautifulSoup) -> dict:
    meta = {}

    title_tag = soup.find("div", class_="articleTitle")
    if title_tag:
        h1 = title_tag.find("h1")
        meta["title"] = clean_text(h1.get_text()) if h1 else ""

    resource_div = soup.find("div", class_="articleResource")
    if resource_div:
        spans = resource_div.find_all("span")
        for span in spans:
            text = clean_text(span.get_text())
            if text.startswith("来源："):
                meta["source"] = text.replace("来源：", "")
            elif text.startswith("时间："):
                meta["date"] = text.replace("时间：", "")
            elif "作者" in text:
                meta["author"] = text.replace("作者：", "").replace("作者", "")

    desc_div = soup.find("div", class_="articleDes")
    if desc_div:
        desc_text = clean_text(desc_div.get_text())
        if desc_text.startswith("摘要："):
            desc_text = desc_text[3:]
        meta["description"] = desc_text

    tags_div = soup.find("div", class_="left2j")
    if tags_div:
        tag_links = tags_div.find_all("a")
        meta["tags"] = [clean_text(a.get_text()) for a in tag_links]

    return meta


def extract_content(soup: BeautifulSoup) -> str:
    """提取正文并转为 Markdown"""
    content_div = soup.find("div", class_="arcContent", id="tupain")
    if not content_div:
        return ""

    outer_table = content_div.find("table", recursive=False)
    if outer_table:
        td = outer_table.find("td")
        if td:
            container = td
        else:
            container = content_div
    else:
        container = content_div

    md_parts: list[str] = []

    for child in container.children:
        if not isinstance(child, Tag):
            continue

        if child.name == "p":
            text = clean_text(child.get_text())
            if not text:
                continue

            style = child.get("style", "")

            has_tip_bg = "background" in style and "border" in style
            is_center = "text-align: center" in style or "text-align:center" in style
            is_right = "text-align: right" in style or "text-align:right" in style

            if has_tip_bg:
                link_text = extract_links(child)
                md_parts.append(f"\n> {link_text}\n")
            elif is_center:
                red_spans = child.find_all("span", style=lambda s: s and "FF0000" in s)
                if red_spans:
                    md_parts.append(f"\n## {text}\n")
                else:
                    blue_spans = child.find_all("span", style=lambda s: s and "0000FF" in s)
                    if blue_spans and len(text) < 60:
                        md_parts.append(f"\n**{text}**\n")
                    else:
                        md_parts.append(f"\n{text}\n")
            elif is_right:
                md_parts.append(f"\n{text}  ")  # trailing spaces for md line break
            else:
                md_parts.append(f"\n{text}\n")

        elif child.name == "table":
            md_parts.append("\n" + html_table_to_narrative(child) + "\n")

    return "\n".join(md_parts)


def build_markdown(meta: dict, content: str) -> str:
    parts = []

    if meta.get("title"):
        parts.append(f"# {meta['title']}\n")

    info_items = []
    if meta.get("date"):
        info_items.append(f"**发布日期：** {meta['date']}")
    if meta.get("source"):
        info_items.append(f"**来源：** {meta['source']}")
    if meta.get("author"):
        info_items.append(f"**作者：** {meta['author']}")
    if info_items:
        parts.append(" | ".join(info_items) + "\n")

    if meta.get("tags"):
        parts.append("**标签：** " + "、".join(meta["tags"]) + "\n")

    if meta.get("description"):
        parts.append(f"> **摘要：** {meta['description']}\n")

    parts.append("---\n")
    parts.append(content)

    return "\n".join(parts)


def main():
    with open(HTML_FILE, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")
    meta = extract_article_meta(soup)
    content = extract_content(soup)
    markdown = build_markdown(meta, content)

    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"解析完成，已输出到: {OUTPUT_FILE}")
    print(f"标题: {meta.get('title', 'N/A')}")
    print(f"日期: {meta.get('date', 'N/A')}")
    print(f"Markdown 长度: {len(markdown)} 字符")


if __name__ == "__main__":
    main()
