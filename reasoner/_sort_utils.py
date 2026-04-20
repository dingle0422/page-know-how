"""跨版本（v0/v1）共享的章节自然排序工具。

放在 `reasoner` 包根而非任一版本目录下，避免版本之间互相 import。
"""

_INF = float("inf")


def natural_dir_sort_key(name: str) -> list:
    """按目录名/章节号的数字前缀做自然排序，避免字典序把
    '10_xxx' 排到 '2_xxx' 之前、或 '2.10' 排到 '2.2' 之前。

    支持三种输入形态：
    - 带尾缀的目录名：'2.10_海外业务' -> [2, 10]
    - 纯章节号：     '2.10'         -> [2, 10]
    - 无数字前缀：   '附录' / 'foo_bar' / 任何含非数字段 -> [inf]，排到末尾
    """
    prefix = name.split("_", 1)[0]
    parts: list = []
    for seg in prefix.split("."):
        try:
            parts.append(int(seg))
        except ValueError:
            return [_INF]
    return parts or [_INF]
