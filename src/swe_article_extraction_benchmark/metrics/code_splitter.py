import re

from swe_article_extraction_benchmark.metrics.base_content_splitter import (
    BaseContentSplitter,
)


class CodeSplitter(BaseContentSplitter):
    def extract(self, text: str) -> str:
        code_blocks = self._extract_basic(text)

        return "\n".join(code_blocks)

    def _extract_basic(self, text: str) -> list[str]:
        code_parts = []

        backtick_pattern = r"(```[\s\S]*?```)"
        for match in re.finditer(backtick_pattern, text):
            code_segment = match.group(0)
            if code_segment.startswith("```"):
                lines = code_segment.split("\n")
                content_lines = lines[1:-1]
                code_content = "\n".join(content_lines)
                if code_content:
                    code_parts.append(code_content)

        indent_pattern = (
            r"(?:\n\s*\n)((?:(?: {4,}|\t+)[^\n]*(?:\n|$)){2,})(?=\n\s*\n|$)"
        )

        for match in re.finditer(indent_pattern, text, re.MULTILINE):
            code_segment = match.group(1)

            # 验证：确保所有行都是缩进的
            lines = code_segment.split("\n")
            all_indented = all(
                line.startswith("    ") or line.startswith("\t") or not line.strip()
                for line in lines
                if line.strip()
            )

            if not all_indented:
                continue

            # 进一步验证代码特征
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            if len(non_empty_lines) < 2:
                continue

            # 检查是否有明显的非代码特征
            has_list_features = any(
                re.match(r"^[-•*]\s", line)
                or re.match(r"^\d+\.\s", line)
                or re.search(r"\$[\d,]", line)
                or re.search(r"\b(million|billion|thousand)\b", line, re.IGNORECASE)
                for line in non_empty_lines
            )

            if has_list_features:
                continue

            # 清理代码段
            cleaned_lines = []
            for line in code_segment.split("\n"):
                if line.strip():
                    if line.startswith("    "):
                        cleaned_lines.append(line[4:])
                    elif line.startswith("\t"):
                        cleaned_lines.append(line[1:])
                    else:
                        cleaned_lines.append(line)

            code_content = "\n".join(cleaned_lines)
            if code_content.strip():
                code_parts.append(code_content)

        return code_parts
