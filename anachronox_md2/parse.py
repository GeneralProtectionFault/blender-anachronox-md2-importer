import os
from pathlib import Path
import json
from dataclasses import dataclass, field, fields
from typing import Any, List
from pyparsing import (
        Word, alphas, alphanums, nums, QuotedString, Suppress, Group, Forward, Literal, Each,
        OneOrMore, ZeroOrMore, Optional, ParseException, ParseResults, pyparsing_common
    )
from pprint import pprint


@dataclass
class Pass:
    map: str = ''
    uvgen: str = ''
    uvmod: str = ''
    blendmode: str = ''
    alphafunc: str = ''
    depthwrite: bool = True
    depthfunc: str = ''
    cull: str = ''
    rgbgen: str = ''

@dataclass
class Skin:
    sort: str = ''
    passes: List[Pass] = field(default_factory=list)

    def add_pass(self, new_pass: Pass):
        if isinstance(new_pass, Pass):
            self.passes.append(new_pass)
        else:
            raise TypeError("Expected an instance of Pass.")

@dataclass
class MDAProfile:
    profile: str = ''
    evaluate: str = ''
    skins: List[Skin] = field(default_factory=list)

    def __iter__(self):
    # iterate over Pass objects across all skins (so comprehensions like
    # [p.map for p in ModelVars.selected_profile] work)
        for skin in self.skins:
            for p in skin.passes:
                yield p

    def __len__(self):
        # total number of Pass objects across all skins
        return sum(len(skin.passes) for skin in self.skins)

    def iter_attr(self, attr_name: str):
        # convenience: iterate the named attribute from each Pass (skips missing/None)
        for p in self:
            val = getattr(p, attr_name, None)
            if val is not None:
                yield val

    def add_skin(self, new_skin: Skin):
        if isinstance(new_skin, Skin):
            self.skins.append(new_skin)
        else:
            raise TypeError("Expected an instance of Skin.")




# Basic punctuation suppressors
LBRACE, RBRACE = map(Suppress, "{}")

# Tokens
# Sometimes the value is in quotes (usually the map), others, it's not, so we need to capture the value in either case
quoted_value = QuotedString('"', escChar="\\", unquoteResults=True)
unquoted_value = Word(alphanums + "._-/\\ ")

quoted_or_unquoted = (quoted_value | unquoted_value)


#########################################################################################################
# pass block: multiple kv pairs inside braces
mda_pass_block = Group(
    Suppress("pass")
    + LBRACE
    + Each(
        Group((Literal("clampmap") | Literal("map")).suppress() + quoted_or_unquoted)("map")
        & Optional(Group(Literal("uvgen").suppress() + Word(alphanums)))("uvgen")
        & Optional(Group(Literal("uvmod").suppress() + Word(alphanums + " .-_")))("uvmod")
        & Optional(Group(Literal("blendmode").suppress() + Word(alphanums)))("blendmode")
        & Optional(Group(Literal("alphafunc").suppress() + Word(alphanums)))("alphafunc")
        & Optional(Group(Literal("depthwrite").suppress() + Word(alphanums)))("depthwrite")
        & Optional(Group(Literal("depthfunc").suppress() + Word(alphanums)))("depthfunc")
        & Optional(Group(Literal("cull").suppress() + Word(alphanums)))("cull")
        & Optional(Group(Literal("rgbgen").suppress() + Word(alphanums)))("rgbgen")
    )
    + RBRACE
)

# skin block: one or more pass blocks
mda_skin_block = Group(
    Suppress("skin")
    + LBRACE
    + Optional(Group(Literal("sort").suppress() + Word(alphas)))("sort")
    + OneOrMore(mda_pass_block)("passes")
    + RBRACE
)

# profile block: 'profile' optionally followed by a token (profile), then braces with skins
mda_profile_block = Group(
    Suppress("profile")
    + Optional(Word(alphanums), default="DFLT")("profile")
    + LBRACE
    + Optional(Group(Literal("evaluate").suppress() + quoted_or_unquoted))("evaluate")
    + OneOrMore(mda_skin_block)("skins")
    + RBRACE
)

#########################################################################################################

atd_header_block = Group(
    Suppress("ATD1") +
    Suppress(Literal("type") + Literal("=")) + Word(alphas)("type") +
    Suppress(Literal("colortype") + Literal("=")) + Word(nums)("colortype") +
    Suppress(Literal("width") + Literal("=")) + Word(nums)("width") +
    Suppress(Literal("height") + Literal("=")) + Word(nums)("height") +
    Suppress(Literal("bilinear") + Literal("=")) + Word(nums)("bilinear")
)

atd_bitmap_block = Group(
    Suppress(Literal("!bitmap") + Literal("file") + Literal("=")) + quoted_or_unquoted
)

atd_frame_block = Group(
    Suppress(Literal("!frame")) +
    Suppress(Literal("bitmap") + Literal("=")) + pyparsing_common.signed_integer("bitmap") +
    Suppress(Literal("next") + Literal("=")) + pyparsing_common.signed_integer("next") +
    Suppress(Literal("wait") + Literal("=")) + (pyparsing_common.signed_integer | pyparsing_common.real)("wait") +
    Optional(Suppress(Literal("x") + Literal("=")) + pyparsing_common.signed_integer)("x") +
    Optional(Suppress(Literal("y") + Literal("=")) + pyparsing_common.signed_integer)("y")
)

atd_block = atd_header_block + OneOrMore(atd_bitmap_block) + OneOrMore(atd_frame_block)

#########################################################################################################


def unwrap_list(value):
    while isinstance(value, (list, ParseResults)) and value:
        value = value[0]
    return value


def parse_mda(filepath):
    # ------------------------------------------------------------------
    # 1.  Grab the whole block that starts with the first “{” and ends
    #     with the matching “}”, but include the line that precedes the
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

    txt = txt[start_index:last_brace+1]

    # Add "outer" brackets, since the profiles themselves are not within brackets - This gives us a consistently formatted string
    # text_to_parse = '{' + text_to_parse + '}'
    txt = txt.replace("\\", "/")    # Fix stupid inconsistent forward/back slashes in MDAs

    # Remove commented lines
    lines = txt.splitlines()
    text_to_parse = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    # print(text_to_parse)

    #####################################################################################################################################

    # top-level: surrounding braces with one or more profiles
    top = OneOrMore(mda_profile_block)("profiles")

    # Parse
    try:
        parsed = top.parseString(text_to_parse, parseAll=True)
        # result = transform(parsed)
        # print(to_json(result))
    except ParseException as pe:
        print("Parse error:", pe)
        raise

    return parsed


def parsed_to_profiles(parsed):
    """
    Convert parsed structure to list[MDAProfile].

    parsed format:
      [
        ['DFLT', [[['map','...'], ['alphafunc','ge128'], ...], [...], ...]],
        ['FLAT', [[['map','...']]]]
      ]

    Behavior:
    - Creates an MDAProfile per top-level entry with profile name set.
    - For each skin entry creates a Skin and for each pass entry creates a Pass.
    - Dynamically assigns key/value pairs to attributes of Pass, Skin, and MDAProfile
      when the attribute name exists in the corresponding dataclass (case-insensitive).
    - Leaves missing attributes as dataclass defaults.
    - Does NOT perform type conversions; values are assigned as strings (or as provided).
    """
    profiles = []
    for profile_item in parsed:
        if not profile_item:
            continue

        profile_name = unwrap_list(profile_item.get("profile"))

        prof = MDAProfile(profile=profile_name)
        prof.evaluate = unwrap_list(profile_item.get("evaluate"))

        skins = profile_item.get("skins")
        for skin_entry in skins:
            # skin_entry is a list of passes (each pass is list of [key,val] pairs)
            skin = Skin()
            skin.sort = unwrap_list(skin_entry.get("sort"))

            passes = skin_entry.get("passes")
            for pass_entry in passes:
                # print(pass_entry)
                p = Pass()
                p.map = unwrap_list(pass_entry.get("map"))
                p.uvgen = unwrap_list(pass_entry.get("uvgen"))
                p.uvmod = unwrap_list(pass_entry.get("uvmod"))
                p.blendmode = unwrap_list(pass_entry.get("blendmode"))
                p.alphafunc = unwrap_list(pass_entry.get("alphafunc"))
                p.depthwrite = unwrap_list(pass_entry.get("depthwrite"))
                p.depthfunc = unwrap_list(pass_entry.get("depthfunc"))
                p.cull = unwrap_list(pass_entry.get("cull"))
                p.rgbgen = unwrap_list(pass_entry.get("rgbgen"))

                # pprint(p.__dict__)
                skin.add_pass(p)
            prof.add_skin(skin)
        profiles.append(prof)

    return profiles



def get_mda_profiles(filepath):
    """Main Function"""
    parsed = parse_mda(filepath)
    return parsed_to_profiles(parsed)


def parse_atd_file(filepath):
    with open(filepath, 'r') as f:
        txt = f.read()

    txt = txt.replace("\\", "/")    # Fix stupid inconsistent forward/back slashes in MDAs

    # Remove commented lines
    lines = txt.splitlines()
    text_to_parse = "\n".join(line for line in lines if not line.lstrip().startswith("#"))

    return atd_block.parseString(text_to_parse, parseAll=True)


if __name__ == "__main__":
    # filepath = '/home/q/ART/Anachronox/MD2_ModelsExtracted/newface/grumpos/grumpos.mda'
    filepath = '/home/q/ART/Anachronox/MD2_ModelsExtracted/newface/boots/skin/boots1_hurt.atd'

    # result = parse_mda(filepath)
    result = parse_atd_file(filepath)

    # Test all MDA files
    # root = Path("/home/q/ART/Anachronox/MD2_ModelsExtracted")

    # mda_files = sorted(root.rglob("*.mda"))
    # for p in mda_files:
    #     print(f"Parsing {p}")
    #     try:
    #         parsed = parse_mda(p)
    #         result = parsed_to_profiles(parsed)
    #     except Exception as e:
    #         print(f"ERROR parsing {p}\n{e}")

    # profiles = parsed_to_profiles(result)
    # for p in profiles:
    #     pprint(p.__dict__)

    print(result)
    print(len(result))

