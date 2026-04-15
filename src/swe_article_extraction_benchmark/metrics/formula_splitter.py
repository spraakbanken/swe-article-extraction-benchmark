import re

from swe_article_extraction_benchmark.metrics.base_content_splitter import (
    BaseContentSplitter,
)


class FormulaSplitter(BaseContentSplitter):
    def extract(
        self,
        text: str,
    ) -> str:
        """提取数学公式"""
        formula_parts = self._extract_basic(text)
        return "\n".join(formula_parts)

    def _extract_basic(self, text: str) -> list[str]:
        """使用正则表达式提取公式"""

        regex_formulas = []

        # 排除Markdown代码块（```code```和`code`）
        # 首先移除多行代码块
        text_without_blocks = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        # 移除内联代码块
        text_without_code = re.sub(r"`[^`]*`", "", text_without_blocks)

        latex_patterns = [
            r"(?<!\\)\$\$(.*?)(?<!\\)\$\$",  # 行间 $$...$$
            r"(?<!\\)\\\[(.*?)(?<!\\)\\\]",  # 行间 \[...\]
            r"(?<!\\)\$(.*?)(?<!\\)\$",  # 行内 $...$
            r"(?<!\\)\\\((.*?)(?<!\\)\\\)",  # 行内 \(...\)
        ]

        for pattern in latex_patterns:
            for match in re.finditer(pattern, text_without_code, re.DOTALL):
                formula_content = match.group(1)
                if formula_content.strip():
                    regex_formulas.append(formula_content.strip())

        return regex_formulas
