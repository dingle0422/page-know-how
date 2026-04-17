# Skills 技能索引

本文件列出可用技能。需要使用某个技能时，加载对应目录下的 `skills.md` 获取详细的调用命令、参数说明、触发场景。

所有技能都以 **bash 命令**形式被调用：直接产出 `python -m skills.<module> ...` 这类命令交给执行环境运行即可。

## 技能注册表

```toml
[standard_product_name_verification]
description = "将商品/服务名称匹配到税收分类编码体系的标准类型名称，返回多个候选项及其标准名称类型、简称、税率和匹配度。"
detail_doc  = "skills/standard_product_name_verification/skills.md"
entry_cmd   = 'python -m skills.standard_product_name_verification "<商品名>" ["<商品名2>" ...] [--json] [--timeout SEC]'
```
