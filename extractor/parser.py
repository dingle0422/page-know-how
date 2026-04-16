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


def _expand_html_table_grid(table_tag) -> list[list[str]]:
    """将 HTML 表格展开为二维文本 grid，正确处理 rowspan/colspan。

    合并单元格的文本会被复制填充到其占据的每一个细单元格中，
    确保展开后的 grid 每行列数一致、数据完整。
    """
    rows = table_tag.find_all('tr')
    if not rows:
        return []

    max_cols = 0
    for row in rows:
        cols_in_row = sum(int(c.get('colspan', 1)) for c in row.find_all(['td', 'th']))
        max_cols = max(max_cols, cols_in_row)

    grid: list[list[str]] = []
    fill: dict[tuple[int, int], str] = {}

    for row_idx, row in enumerate(rows):
        cells = row.find_all(['td', 'th'])
        grid_row = [''] * max_cols
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
            text = cell.get_text(strip=True).replace('\n', '；')
            rs = int(cell.get('rowspan', 1))
            cs = int(cell.get('colspan', 1))

            for c in range(cs):
                grid_row[col_pos + c] = text
            for r in range(1, rs):
                for c in range(cs):
                    fill[(row_idx + r, col_pos + c)] = text

            cell_idx += 1
            col_pos += cs

        grid.append(grid_row)

    return grid


def _build_narrative_headers(raw_header_row: list[str]) -> tuple[list[str], list[int]]:
    """从展开后的表头行构建去重的列名列表。

    - 相邻重复的列名（由 colspan 展开产生）以"列名1、列名2…"区分。
    - 返回 (seen_headers, seen_indices)，seen_indices 覆盖所有有效列。
    """
    headers: list[str] = []
    for h in raw_header_row:
        cleaned = re.sub(r'\s+', '', h)
        headers.append(cleaned)

    name_positions: dict[str, list[int]] = {}
    for i, h in enumerate(headers):
        if h:
            name_positions.setdefault(h, []).append(i)

    final_names: list[str] = [''] * len(headers)
    for name, positions in name_positions.items():
        if len(positions) == 1:
            final_names[positions[0]] = name
        else:
            for seq, pos in enumerate(positions, start=1):
                final_names[pos] = f"{name}{seq}"

    seen_headers: list[str] = []
    seen_indices: list[int] = []
    for i, h in enumerate(final_names):
        if h and h not in seen_headers:
            seen_headers.append(h)
            seen_indices.append(i)

    return seen_headers, seen_indices


def _html_table_to_narrative(table_tag) -> str:
    """将单个 HTML <table> 标签转为口语化编号列表，每条数据以（1）（2）… 列出。

    正确处理 rowspan/colspan：合并单元格的文本填充到所有占据的位置。
    表头若存在横向合并（colspan），展开为"表头名1、表头名2…"。
    """
    grid = _expand_html_table_grid(table_tag)
    if not grid:
        return ''

    seen_headers, seen_indices = _build_narrative_headers(grid[0])

    if not seen_headers:
        return ''

    if len(grid) <= 1:
        return "表头：" + "、".join(seen_headers)

    lines: list[str] = []
    for row_idx, row in enumerate(grid[1:], start=1):
        parts: list[str] = []
        for i, col_idx in enumerate(seen_indices):
            header = seen_headers[i]
            value = row[col_idx] if col_idx < len(row) else ''
            if value:
                parts.append(f"{header}：{value}")
        if parts:
            lines.append(f"（{row_idx}）{'；'.join(parts)}。")

    return '\n'.join(lines)


def _convert_html_to_markdown(text: str, narrativize: bool = True) -> str:
    """将文本中的 HTML 片段转为 Markdown，纯文本原样返回。

    Args:
        text: 原始文本（可能包含 HTML）
        narrativize: 若为 True，将 HTML 表格转为口语化编号列表（（1）（2）…），
                     结合字段名称描述每条数据，便于 LLM 理解；
                     若为 False，使用常规 Markdown 表格格式。
    """
    if not HTML_TAG_RE.search(text):
        return text

    from markdownify import markdownify as md

    if not narrativize:
        return md(text, strip=['span', 'div', 'colgroup', 'col']).strip()

    from bs4 import BeautifulSoup, NavigableString

    soup = BeautifulSoup(text, 'html.parser')
    tables = soup.find_all('table')

    if not tables:
        return md(text, strip=['span', 'div', 'colgroup', 'col']).strip()

    for table in tables:
        narrative = _html_table_to_narrative(table)
        table.replace_with(NavigableString(narrative))

    return md(str(soup), strip=['span', 'div', 'colgroup', 'col']).strip()


def _derive_parent_number(number: str) -> str:
    """
    从层级编号推导其直接父级编号。
    例如 '1.2.1' → '1.2'，'1.2' → '1'，'1' → ''，'前言' → ''
    """
    parts = number.split('.')
    if len(parts) <= 1:
        return ''
    return '.'.join(parts[:-1])


def _build_full_path(number: str, node_map: dict[str, dict]) -> str:
    """
    从 number 向上遍历父级链，构建完整祖先路径。
    返回格式如 '1/1.2/1.2.1'（不含自身编号）。
    """
    ancestors: list[str] = []
    current = number
    while True:
        parent_num = _derive_parent_number(current)
        if not parent_num or parent_num not in node_map:
            break
        ancestors.append(parent_num)
        current = parent_num
    ancestors.reverse()
    return '/'.join(ancestors)


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
    - path 字段自动补全：原始 path 只含直接父级编号，自动构建完整祖先路径
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    clause_list: list[dict] = data.get('response', {}).get('clause_list', [])
    if not clause_list:
        raise ValueError(f"JSON 文件中未找到 response.clause_list: {filepath}")

    number_set: dict[str, dict] = {raw.get('number', ''): raw for raw in clause_list}

    result: list[Clause] = []
    for raw in clause_list:
        number = raw.get('number', '')
        if number == '前言':
            number = '0'
        raw_path = raw.get('path', '')

        if raw_path:
            path = _build_full_path(number, number_set)
        else:
            path = ''

        clause: Clause = {
            'number': number,
            'content': _convert_html_to_markdown(raw.get('content', '')),
            'level': raw.get('level', 0),
            'path': path,
            'full_name': raw.get('full_name', ''),
        }
        result.append(clause)

    logger.info(f"从 JSON 解析到 {len(result)} 个条款（已完成 HTML→Markdown 转换与路径补全）")
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


DEFAULT_API_URL = "http://10.199.0.40:8080/kg-platform/api/clauses/list"


def _extract_version_from_clause_name(clause_name: str) -> str:
    """从 clauseName 中提取括号内的版本号，如 '（20260413110049）' → '20260413110049'"""
    m = re.search(r'[（(](\d+)[）)]', clause_name)
    return m.group(1) if m else ''


def _extract_policy_name(clause_name: str) -> str:
    """从 clauseName 中提取策略名称（去掉 'clauseNumber_' 前缀和版本号后缀）"""
    name = re.sub(r'^[^_]+_', '', clause_name, count=1)
    name = re.sub(r'[（(]\d+[）)]', '', name).strip()
    return name


def fetch_api_clauses(
    policy_id: str,
    api_url: str = DEFAULT_API_URL,
) -> tuple[list[Clause], str, str, dict[str, dict]]:
    """
    从 API 接口获取条款数据并转换为 Clause 列表。

    参数:
        policy_id: 策略 ID
        api_url: API 接口地址

    返回:
        (clauses, policy_name, version, raw_clause_map)
        - clauses: 转换后的 Clause 列表（与 parse_clause_json 格式一致）
        - policy_name: 从 clauseName 提取的策略名称
        - version: 从 clauseName 提取的版本号（用作目录时间戳后缀）
        - raw_clause_map: number → 原始 API 条款数据的映射（用于保存 clause.json）
    """
    import requests

    logger.info(f"从 API 获取条款数据: policyId={policy_id}, url={api_url}")
    response = requests.post(api_url, json={"policyId": policy_id})
    response.raise_for_status()

    result = response.json()
    if not result.get('success'):
        raise ValueError(f"API 返回失败: {result.get('message', '未知错误')}")

    api_clauses: list[dict] = result.get('data', {}).get('clauses', [])
    if not api_clauses:
        raise ValueError(f"API 未返回有效条款数据，policyId: {policy_id}")

    policy_name = ''
    version = ''
    for c in api_clauses:
        cn = c.get('clauseName', '')
        v = _extract_version_from_clause_name(cn)
        if v:
            version = v
            if not policy_name:
                policy_name = _extract_policy_name(cn)
            break

    if not policy_name:
        policy_name = policy_id

    number_set: dict[str, dict] = {}
    raw_clause_map: dict[str, dict] = {}
    clauses: list[Clause] = []

    for raw in api_clauses:
        number = raw.get('clauseNumber', '')
        if number == '前言':
            number = '0'

        number_set[number] = raw
        raw_clause_map[number] = raw

        search_labels = raw.get('searchLabels', [])
        full_name = search_labels[0] if search_labels else ''

        content = _convert_html_to_markdown(raw.get('clauseContent', ''))
        level = int(raw.get('level', 0))

        clause: Clause = {
            'number': number,
            'content': content,
            'level': level,
            'path': '',
            'full_name': full_name,
        }
        clauses.append(clause)

    for clause in clauses:
        number = clause['number']
        parent_num = _derive_parent_number(number)
        if parent_num and parent_num in number_set:
            clause['path'] = _build_full_path(number, number_set)

    logger.info(f"从 API 获取到 {len(clauses)} 个条款（policyId: {policy_id}，版本: {version}）")
    return clauses, policy_name, version, raw_clause_map
