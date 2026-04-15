def extract_title(html: str) -> str | None:
    """Extract title from html."""
    try:
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL
        )
        if title_match:
            return title_match.group(1).strip()
    except Exception:
        pass
    return None


def detect_language(content: str | None) -> str | None:
    """Detect language in content."""
    if not content:
        return None

    # 简单的语言检测逻辑
    english_chars = len(re.findall(r"[a-zA-Z]", content))
    swedish_chars = len(re.findall(r"och|[åäö]", content))

    if swedish_chars > english_chars:
        return "swe"
    elif english_chars > 0:
        return "eng"
    else:
        return None
