import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import TypedDict

logger = logging.getLogger(__name__)

HTML_TAG_RE = re.compile(r'<[a-zA-Z][^>]*>')


class Clause(TypedDict):
    number: str
    content: str
    level: int
    path: str
    full_name: str


@dataclass
class ParsedLine:
    text: str
    is_heading_style: bool = False
    heading_level: int = 0  # 0 = 非标题；1/2/3... = Word Heading 层级


def _table_to_narrative(table) -> str:
    """
    将 docx 表格转为口语化的文字描述。
    以表头为 key，逐行将每个单元格内容以"字段名：xxx"的方式串联。
    """
    rows = table.rows
    if len(rows) == 0:
        return ""

    headers = [cell.text.strip().replace('\n', ' ') for cell in rows[0].cells]
    seen_headers = []
    seen_indices = []
    for i, h in enumerate(headers):
        if h not in seen_headers:
            seen_headers.append(h)
            seen_indices.append(i)

    if len(rows) == 1:
        return "【表格】表头：" + "、".join(seen_headers)

    narrative_lines = ["【表格内容】"]
    for row_idx, row in enumerate(rows[1:], start=1):
        cells = [cell.text.strip().replace('\n', '；') for cell in row.cells]
        parts = []
        for i, col_idx in enumerate(seen_indices):
            header = seen_headers[i]
            value = cells[col_idx] if col_idx < len(cells) else ""
            if value:
                parts.append(f"{header}：{value}")
        if parts:
            narrative_lines.append(f"  第{row_idx}项：{'；'.join(parts)}。")

    return "\n".join(narrative_lines)


def _is_heading_style(style_name: str | None) -> bool:
    """判断段落样式是否为 Heading 样式"""
    if not style_name:
        return False
    name = style_name.lower()
    if name.startswith('heading') or name.startswith('标题'):
        return True
    if name.startswith('toc'):
        return False
    return False


def _get_heading_level(style_name: str | None) -> int:
    """从 Word 标题样式名提取层级数字，例如 'Heading 1' / '标题1' → 1；非标题 → 0"""
    if not style_name:
        return 0
    name = style_name.lower().strip()
    if name.startswith('heading') or name.startswith('标题'):
        m = re.search(r'(\d+)\s*$', name)
        if m:
            return int(m.group(1))
    return 0


def parse_docx(filepath: str) -> list[ParsedLine]:
    """
    解析 docx 文件，返回结构化行数据。
    每行标记是否来自 Heading 样式的段落，供下游做两层识别：
      1) 有 Heading 样式 → 再检查文本是否匹配 1./1.1./1.1.1. 格式
      2) 无 Heading 样式 → 视为普通内容
    表格以口语化方式转译。
    """
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(filepath)
    body = doc.element.body

    table_elements = [tbl._element for tbl in doc.tables]
    table_map = {id(el): tbl for el, tbl in zip(table_elements, doc.tables)}

    para_map = {}
    for para in doc.paragraphs:
        para_map[id(para._element)] = para

    result: list[ParsedLine] = []

    for child in body:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag

        if tag == 'p':
            texts = [node.text for node in child.iter(qn('w:t')) if node.text]
            para_text = ''.join(texts)

            para_obj = para_map.get(id(child))
            heading = False
            hlevel = 0
            if para_obj is not None:
                style_name = para_obj.style.name if para_obj.style else None
                heading = _is_heading_style(style_name)
                hlevel = _get_heading_level(style_name)

            result.append(ParsedLine(text=para_text, is_heading_style=heading, heading_level=hlevel))

        elif tag == 'tbl':
            tbl_obj = table_map.get(id(child))
            if tbl_obj is None:
                tbl_obj = Table(child, doc)
            narrative = _table_to_narrative(tbl_obj)
            if narrative:
                for line in narrative.split('\n'):
                    result.append(ParsedLine(text=line, is_heading_style=False))

    return result


def parse_txt(filepath: str) -> list[ParsedLine]:
    """解析 txt 文件，自动检测编码（utf-8 优先，gbk 兜底）。无样式信息。"""
    for encoding in ("utf-8", "gbk", "utf-8-sig", "latin-1"):
        try:
            with open(filepath, "r", encoding=encoding) as f:
                content = f.read()
            return [ParsedLine(text=line, is_heading_style=False) for line in content.split('\n')]
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法识别文件编码: {filepath}")


def _convert_html_to_markdown(text: str) -> str:
    """将文本中的 HTML 片段转为 Markdown，纯文本原样返回"""
    if not HTML_TAG_RE.search(text):
        return text
    from markdownify import markdownify as md
    return md(text, strip=['span', 'div', 'colgroup', 'col']).strip()


def parse_clause_json(filepath: str) -> list[Clause]:
    """
    解析 clause_list JSON 文件，返回清洗后的条款列表。

    输入 JSON 结构示例::

        {
          "response": {
            "clause_list": [
              {"number": "1", "content": "...", "level": 1, "path": "", "full_name": "..."},
              ...
            ]
          }
        }

    处理逻辑：
    - 从 response.clause_list 提取条款数组
    - 将 content 字段中的 HTML（Quill 编辑器产出的表格等）转为 Markdown
    - content 中的编号前缀（如 "1.2.1.  "）已在原始数据中存在，保持原样
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    clause_list: list[dict] = data.get('response', {}).get('clause_list', [])
    if not clause_list:
        raise ValueError(f"JSON 文件中未找到 response.clause_list: {filepath}")

    result: list[Clause] = []
    for raw in clause_list:
        clause: Clause = {
            'number': raw.get('number', ''),
            'content': _convert_html_to_markdown(raw.get('content', '')),
            'level': raw.get('level', 0),
            'path': raw.get('path', ''),
            'full_name': raw.get('full_name', ''),
        }
        result.append(clause)

    logger.info(f"从 JSON 解析到 {len(result)} 个条款（已完成 HTML→Markdown 转换）")
    return result


def parse_document(filepath: str) -> list[ParsedLine]:
    """统一文档解析入口，根据扩展名分派到对应解析器"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".docx":
        logger.info(f"解析 docx 文件: {filepath}")
        return parse_docx(filepath)
    elif ext == ".txt":
        logger.info(f"解析 txt 文件: {filepath}")
        return parse_txt(filepath)
    else:
        raise ValueError(f"不支持的文件类型: {ext}，仅支持 .docx 和 .txt")
