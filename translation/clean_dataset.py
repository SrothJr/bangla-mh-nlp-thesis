"""
clean_dataset.py — Pre-translation dataset cleaner for merged_mental_health_dapt.jsonl

Identified issues (271,664 total rows):
  1. URLs (1.6%)          — raw http/www links confuse the MT model; strip them.
  2. Duplicates (7.7%)    — exact-duplicate posts waste compute; deduplicate.
  3. HTML entities (6.3%) — &amp; &lt; &gt; &quot; etc. -> decode to real chars.
  4. Markdown bold/italic (4.7%) — **text** / *text* / __text__ -> strip markers.
  5. Emoji (1.7%)         — non-BMP Unicode emoji -> remove (untranslatable noise).
  6. Markdown hyperlinks (1.0%) — [label](url) -> keep label, discard URL.
  7. Subreddit mentions (1.2%) — r/SubredditName -> keep as-is (English proper nouns;
                                 masking handles [Subreddit:X] prefix separately).
  8. User mentions (0.1%) — u/username -> remove entirely (PII + untranslatable).
  9. Garbled/replacement chars (0.1%) — mojibake U+FFFD -> strip.
 10. Markdown blockquotes (0.1%) — leading "> " -> strip the ">" marker.
 11. Markdown strikethrough (0.1%) — ~~text~~ -> strip markers, keep text.
 12. Markdown headers (0.0%) — ## heading -> strip "#" markers.
 13. Excessive whitespace (0.1%) — 3+ newlines -> 2 newlines; 3+ spaces -> 1 space.
 14. Very long posts (10.3%) — posts > 400 words hit NLLB's 1024-token limit and
                               get silently truncated. We CHUNK these into <=400-word
                               segments stored as a list in a new field `text_chunks`.
                               The translator must process each chunk separately.
                               (Short posts keep text_chunks = [text] for uniformity.)

What is deliberately NOT changed:
  - [Subreddit: xxx] prefix — used downstream by the pipeline; left alone.
  - Protected terms (medicine names, acronyms) — handled by mask_utils.py.
  - Subreddit mentions (r/xxx) — treated as proper nouns; left for the model.
  - Intentional punctuation, slang, spelling errors — these carry meaning.

Output:
  - cleaned_mental_health_dapt.jsonl   — one row per original post (deduped),
                                         with cleaned `text` and `text_chunks` fields.
  - cleaning_report.json               — summary statistics of what was changed.

Usage:
    python clean_dataset.py
    python clean_dataset.py --input merged_mental_health_dapt.jsonl \\
                            --output cleaned_mental_health_dapt.jsonl \\
                            --report cleaning_report.json \\
                            --chunk_words 400
"""

import argparse
import html
import json
import re
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# URLs: http/https/www
_URL = re.compile(r'https?://\S+|www\.\S+')

# Markdown hyperlinks [label](url) — capture the label, drop the URL
_MD_LINK = re.compile(r'\[([^\]\n]+)\]\(https?://[^)]*\)')

# Markdown bold/italic: ***text***, **text**, *text*, ___text___, __text__, _text_
# Non-greedy inner match to avoid crossing sentence boundaries.
_MD_BOLD_ITALIC = re.compile(
    r'(\*{1,3}|_{1,3})(.+?)\1',
    re.DOTALL
)

# Markdown strikethrough ~~text~~
_MD_STRIKE = re.compile(r'~~(.+?)~~', re.DOTALL)

# Markdown blockquote: leading ">" (possibly multiple) at start of line
_MD_BLOCKQUOTE = re.compile(r'^>+\s?', re.MULTILINE)

# Markdown headers: # ## ### etc. at start of line
_MD_HEADER = re.compile(r'^#{1,6}\s+', re.MULTILINE)

# User mentions u/username (case-insensitive)
_USER_MENTION = re.compile(r'\bu/\w+', re.IGNORECASE)

# Unicode replacement character U+FFFD and null bytes
_REPLACEMENT_CHAR = re.compile(r'[\ufffd\x00]')

# Non-BMP emoji (U+10000 and above — most common emoji block)
_EMOJI = re.compile(r'[\U00010000-\U0010FFFF]', flags=re.UNICODE)

# Excessive newlines (3+)
_MULTI_NL = re.compile(r'\n{3,}')

# Excessive spaces (3+, not newlines)
_MULTI_SP = re.compile(r'[^\S\n]{3,}')

# Trailing whitespace on each line
_TRAILING_WS = re.compile(r'[ \t]+$', re.MULTILINE)

# HTML entity detection (for pre-clean counting)
_HTML_ENTITY = re.compile(r'&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);',
                           re.IGNORECASE)

# Mojibake: UTF-8 bytes decoded as Latin-1 — common patterns seen in the data.
# These are multi-byte sequences that appear as garbled chars when text was
# originally UTF-8 but was mis-decoded as Windows-1252 or Latin-1 at some point.
_MOJIBAKE_MAP = [
    ('\u00e2\u0080\u0099', '\u2019'),   # a€™  ->  ' (right single quote)
    ('\u00e2\u0080\u0098', '\u2018'),   # a€˜  ->  ' (left single quote)
    ('\u00e2\u0080\u009c', '\u201c'),   # a€œ  ->  " (left double quote)
    ('\u00e2\u0080\u009d', '\u201d'),   # a€   ->  " (right double quote)
    ('\u00e2\u0080\u0094', '\u2014'),   # a€"  ->  — (em dash)
    ('\u00e2\u0080\u0093', '\u2013'),   # a€"  ->  – (en dash)
    ('\u00e2\u0080\u00a6', '\u2026'),   # a€¦  ->  … (ellipsis)
]

# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

def fix_mojibake(text: str) -> str:
    """Fix common double-encoding artifacts before any other cleaning."""
    for bad, good in _MOJIBAKE_MAP:
        if bad in text:
            text = text.replace(bad, good)
    return text


def decode_html_entities(text: str) -> str:
    """Decode HTML entities: &amp; -> &, &lt; -> <, &#39; -> ', etc."""
    return html.unescape(text)


def strip_urls(text: str) -> str:
    """Remove raw URLs from text."""
    return _URL.sub('', text)


def fix_markdown_links(text: str) -> str:
    """Replace [label](url) with just the label text."""
    return _MD_LINK.sub(r'\1', text)


def strip_markdown_formatting(text: str) -> str:
    """Strip bold/italic/strikethrough/blockquote/header markers, keeping inner text."""
    # Headers: remove # prefix, keep heading text
    text = _MD_HEADER.sub('', text)
    # Strikethrough: keep inner text
    text = _MD_STRIKE.sub(r'\1', text)
    # Bold/italic: keep inner text (handle nested like ***bold italic***)
    # Iterate a few times because nested markers may need multiple passes
    for _ in range(3):
        text = _MD_BOLD_ITALIC.sub(r'\2', text)
    # Blockquote markers: strip the leading >
    text = _MD_BLOCKQUOTE.sub('', text)
    return text


def remove_user_mentions(text: str) -> str:
    """Remove u/username references (PII + meaningless to translator)."""
    return _USER_MENTION.sub('', text)


def remove_emoji(text: str) -> str:
    """Remove non-BMP emoji characters."""
    return _EMOJI.sub('', text)


def remove_replacement_chars(text: str) -> str:
    """Remove Unicode replacement character U+FFFD and null bytes."""
    return _REPLACEMENT_CHAR.sub('', text)


def normalize_whitespace(text: str) -> str:
    """Normalize excessive newlines and spaces, strip trailing whitespace per line."""
    text = _TRAILING_WS.sub('', text)
    text = _MULTI_NL.sub('\n\n', text)
    text = _MULTI_SP.sub(' ', text)
    return text.strip()


def clean_text(text: str) -> tuple:
    """
    Apply all cleaning steps in safe order.
    Returns (changes_list, cleaned_text).
    changes_list is a list of issue keys that were found and fixed.
    """
    changes = []

    # Step 1: Fix mojibake FIRST (before any regex that might see weird chars)
    t = fix_mojibake(text)
    if t != text:
        changes.append('mojibake_fixed')
    text = t

    # Step 2: Decode HTML entities
    t = decode_html_entities(text)
    if t != text:
        changes.append('html_entity_decoded')
    text = t

    # Step 3: Markdown hyperlinks — extract label, drop URL
    t = fix_markdown_links(text)
    if t != text:
        changes.append('markdown_link_fixed')
    text = t

    # Step 4: Strip raw URLs
    t = strip_urls(text)
    if t != text:
        changes.append('url_stripped')
    text = t

    # Step 5: Strip markdown formatting (bold, italic, strike, blockquote, headers)
    t = strip_markdown_formatting(text)
    if t != text:
        changes.append('markdown_stripped')
    text = t

    # Step 6: Remove u/username mentions
    t = remove_user_mentions(text)
    if t != text:
        changes.append('user_mention_removed')
    text = t

    # Step 7: Remove emoji
    t = remove_emoji(text)
    if t != text:
        changes.append('emoji_removed')
    text = t

    # Step 8: Remove replacement/garbled chars
    t = remove_replacement_chars(text)
    if t != text:
        changes.append('replacement_char_removed')
    text = t

    # Step 9: Normalize whitespace
    t = normalize_whitespace(text)
    if t != text:
        changes.append('whitespace_normalized')
    text = t

    return changes, text


def chunk_text(text: str, max_words: int = 400) -> list:
    """
    Split text into chunks of at most max_words words, breaking on paragraph
    boundaries (double newlines) where possible, falling back to sentence
    boundaries ('. ') if a single paragraph is still too long.

    The [Subreddit: xxx] prefix is intentional — it is prepended to EVERY chunk
    so each segment has full context when fed to the translation model.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    # Extract the [Subreddit: xxx] prefix if present
    prefix = ''
    body = text
    prefix_match = re.match(r'^(\[Subreddit:[^\]]+\])\s*', text)
    if prefix_match:
        prefix = prefix_match.group(0).rstrip()  # e.g. "[Subreddit: addiction]"
        body = text[len(prefix_match.group(0)):]

    # Split body on paragraph breaks
    paragraphs = re.split(r'\n\n+', body.strip())

    chunks = []
    current_words = []

    for para in paragraphs:
        para_words = para.split()

        # If a single paragraph is itself too long, split it on sentence ends
        if len(para_words) > max_words:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_words = sent.split()
                if len(current_words) + len(sent_words) > max_words and current_words:
                    chunk_str = (' '.join(current_words)).strip()
                    # Prefix goes on every chunk
                    if prefix:
                        chunk_str = prefix + ' ' + chunk_str
                    chunks.append(chunk_str)
                    current_words = []
                current_words.extend(sent_words)
        else:
            if len(current_words) + len(para_words) > max_words and current_words:
                chunk_str = (' '.join(current_words)).strip()
                if prefix:
                    chunk_str = prefix + ' ' + chunk_str
                chunks.append(chunk_str)
                current_words = []
            current_words.extend(para_words)

    if current_words:
        chunk_str = (' '.join(current_words)).strip()
        if prefix:
            chunk_str = prefix + ' ' + chunk_str
        chunks.append(chunk_str)

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Pre-translation cleaner for mental health JSONL dataset.'
    )
    ap.add_argument('--input',  default='merged_mental_health_dapt.jsonl')
    ap.add_argument('--output', default='cleaned_mental_health_dapt.jsonl')
    ap.add_argument('--report', default='cleaning_report.json')
    ap.add_argument(
        '--chunk_words', type=int, default=400,
        help='Max words per text chunk (NLLB safe limit). Default: 400'
    )
    ap.add_argument(
        '--no_dedup', action='store_true',
        help='Skip duplicate removal (keep all rows)'
    )
    args = ap.parse_args()

    issue_counter  = Counter()
    change_counter = Counter()
    total_in       = 0
    total_out      = 0
    skipped_empty  = 0
    skipped_dup    = 0
    chunked_rows   = 0
    total_chunks   = 0

    seen_texts: set = set()  # for deduplication (first 300 chars of cleaned text)

    print(f'Reading:  {args.input}')
    print(f'Writing:  {args.output}')
    print(f'Report:   {args.report}')
    print(f'Max words per chunk: {args.chunk_words}')
    print()

    with open(args.input,  encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:

        for line_no, raw_line in enumerate(fin):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            total_in += 1

            # -- Parse JSON ------------------------------------------------
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as e:
                issue_counter['bad_json'] += 1
                print(f'  [SKIP] line {line_no}: bad JSON -- {e}', file=sys.stderr)
                continue

            text = obj.get('text', '')
            if not isinstance(text, str) or not text.strip():
                skipped_empty += 1
                issue_counter['empty_text'] += 1
                continue

            # -- Pre-clean issue counting (for report) ---------------------
            if _URL.search(text):               issue_counter['has_url'] += 1
            if _HTML_ENTITY.search(text):       issue_counter['has_html_entity'] += 1
            if _MD_BOLD_ITALIC.search(text):    issue_counter['has_markdown_formatting'] += 1
            if _MD_LINK.search(text):           issue_counter['has_markdown_link'] += 1
            if _USER_MENTION.search(text):      issue_counter['has_user_mention'] += 1
            if _EMOJI.search(text):             issue_counter['has_emoji'] += 1
            if _REPLACEMENT_CHAR.search(text):  issue_counter['has_replacement_char'] += 1
            if _MULTI_NL.search(text):          issue_counter['has_excessive_newlines'] += 1
            if len(text.split()) > args.chunk_words:
                issue_counter['very_long_text'] += 1

            # -- Clean text ------------------------------------------------
            changes, cleaned = clean_text(text)
            for c in changes:
                change_counter[c] += 1

            if not cleaned.strip():
                skipped_empty += 1
                issue_counter['empty_after_clean'] += 1
                continue

            # -- Deduplication --------------------------------------------
            if not args.no_dedup:
                dedup_key = cleaned.strip()[:300]
                if dedup_key in seen_texts:
                    skipped_dup += 1
                    issue_counter['duplicate_skipped'] += 1
                    continue
                seen_texts.add(dedup_key)

            # -- Chunking --------------------------------------------------
            chunks = chunk_text(cleaned, max_words=args.chunk_words)
            if len(chunks) > 1:
                chunked_rows += 1
                total_chunks += len(chunks)
            else:
                total_chunks += 1

            # -- Build output row ------------------------------------------
            out_obj = dict(obj)
            out_obj['text']        = cleaned    # cleaned full text
            out_obj['text_chunks'] = chunks     # list of <=400-word segments
            out_obj['source_line'] = line_no    # 0-indexed line in original file
            if changes:
                out_obj['clean_fixes'] = changes

            fout.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
            total_out += 1

            if total_in % 20000 == 0:
                print(f'  Processed {total_in:,} rows -> kept {total_out:,} so far ...')

    # -- Report -----------------------------------------------------------
    report = {
        'input_file':              args.input,
        'output_file':             args.output,
        'total_rows_read':         total_in,
        'rows_output':             total_out,
        'skipped_empty':           skipped_empty,
        'skipped_duplicate':       skipped_dup,
        'rows_chunked':            chunked_rows,
        'total_chunks_produced':   total_chunks,
        'issues_found_pre_clean':  dict(issue_counter),
        'fixes_applied':           dict(change_counter),
        'chunk_word_limit':        args.chunk_words,
    }

    with open(args.report, 'w', encoding='utf-8') as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)

    print()
    print('=== CLEANING COMPLETE ===')
    print(f'  Total rows read:         {total_in:,}')
    print(f'  Rows written (output):   {total_out:,}')
    print(f'  Skipped (empty):         {skipped_empty:,}')
    print(f'  Skipped (duplicate):     {skipped_dup:,}')
    print(f'  Rows split into chunks:  {chunked_rows:,}')
    print(f'  Total chunks produced:   {total_chunks:,}')
    print()
    print('Issues found (pre-clean):')
    for k, v in sorted(issue_counter.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v:,}')
    print()
    print('Fixes applied:')
    for k, v in sorted(change_counter.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v:,}')
    print()
    print(f'Report saved to: {args.report}')


if __name__ == '__main__':
    main()
