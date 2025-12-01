import json
from typing import Any
from pyparsing import (
        Word, alphas, alphanums, QuotedString, Suppress, Group, Forward,
        OneOrMore, ZeroOrMore, Optional, ParseException
    )


def to_json(result: Any, pretty: bool = False) -> str:
    """
    Convert parsed result (list/dicts) to a JSON string.
    Set pretty=True for indented, human-readable output.
    """
    if pretty:
        return json.dumps(result, indent=2, ensure_ascii=False)
    return json.dumps(result, separators=(",", ":"), ensure_ascii=False)


def parse_mda(filepath):
    # ------------------------------------------------------------------
    # 1.  Grab the whole block that starts with the first “{” and ends
    #     with the matching “}”, but include the word that precedes the
    #     first “{” – that is the first key (e.g. “profile”).
    # ------------------------------------------------------------------

    with open(filepath, 'r') as f:
        txt = f.read()

    # Find the first “{” and the word right before it
    first_brace = txt.find('{')
    last_brace = txt.rfind('}')
    if first_brace == -1 or last_brace == -1:
        raise ValueError('No/inconsistent braces found in file')

    # Get the index of the beginning of the line just before the first curly brace
    # This is to capture:
    # profile
    # {....
    line_start_index = txt.rfind('\n', 0, first_brace) + 1
    # (previous line)
    start_index = txt.rfind('\n', 0, line_start_index - 1) + 1

    text_to_parse = txt[start_index:last_brace+1]

    # Add "outer" brackets, since the profiles themselves are not within brackets - This gives us a consistently formatted string
    text_to_parse = '{' + text_to_parse + '}'
    text_to_parse = text_to_parse.replace("\\", "/")    # Fix stupid inconsistent forward/back slashes in MDAs
    # print(text_to_parse)

    #####################################################################################################################################

    # Basic punctuation suppressors
    LBRACE, RBRACE = map(Suppress, "{}")

    # Tokens
    identifier = Word(alphas + "_", alphanums + "_")
    # quoted strings: QuotedString handles quotes properly and removeQuotes strips the quotes
    quoted_value = QuotedString('"', escChar="\\", unquoteResults=True)
    # unquoted values: include letters, numbers, dots, slashes, backslashes, hyphen, underscore
    unquoted_value = Word(alphanums + "._-/\\")

    value = (quoted_value | unquoted_value)

    # key and kv pair (key followed by one value on the same line)
    key = Word(alphas + "_")
    kv_pair = Group(key + value)

    # pass block: multiple kv pairs inside braces
    pass_block = Group(Suppress("pass") + LBRACE + ZeroOrMore(kv_pair) + RBRACE)

    # skin block: one or more pass blocks
    skin_block = Group(Suppress("skin") + LBRACE + OneOrMore(pass_block) + RBRACE)

    # profile block: 'profile' optionally followed by a token (modifier), then braces with skins
    profile_block = Group(Suppress("profile") + Optional(Word(alphanums + "_-"))("modifier") + LBRACE + OneOrMore(skin_block)("skins") + RBRACE)

    # top-level: surrounding braces with one or more profiles
    top = LBRACE + OneOrMore(profile_block)("profiles") + RBRACE

    # helper to turn kv pairs into dict
    def make_pass_dict(kv_tokens):
        d = {}
        for kv in kv_tokens:
            k = kv[0]
            v = kv[1]
            d[k] = v
        return d

    # transform parse results to desired structure
    def transform(parsed):
        out = []
        for prof in parsed.profiles:
            prof_name = prof.get("modifier") or "EMPTY"
            skins = []
            for skin in prof.skins:
                passes = []
                for p in skin:          # each p is a list of kv pairs
                    passes.append(make_pass_dict(p))
                skins.append({"passes": passes})
            out.append({"profile": prof_name, "skins": skins})
        return out

    # Parse
    try:
        parsed = top.parseString(text_to_parse, parseAll=True)
        result = transform(parsed)
        # print(to_json(result))
    except ParseException as pe:
        print("Parse error:", pe)
        raise

    return result


if __name__ == "__main__":
    filepath = '/home/q/ART/Anachronox/MD2_ModelsExtracted/pal/pal.mda'
    result = parse_mda(filepath)

    # print(result)
    print(len(result))
    # print(to_json(result))

    selected_profile = [r for r in result if r['profile'] == "EMPTY"]
    # print([r for r in result if r['profile'] == "EMPTY"])

    maps = [p['map']
        for profile in selected_profile
        for skin in profile.get('skins', [])
        for p in skin.get('passes', [])
        if 'map' in p]

    print(isinstance(maps, list))