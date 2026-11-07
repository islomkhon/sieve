"""Deterministic surface normalization for broad English phrasing.

These rules cover contractions, polite requests, common phrasal verbs, and
paraphrases of the same question. They rewrite surface forms into the small
canonical grammar; they do not supply word meanings.
"""

from __future__ import annotations

import re


CONTRACTIONS = {
    "ain't": "is not",
    "can't": "can not",
    "cannot": "can not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "here's": "here is",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "when's": "when is",
    "where's": "where is",
    "who's": "who is",
    "why's": "why is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    # Apostrophes are commonly omitted in informal typing. These entries are
    # unambiguous enough to normalize before the parser sees them.
    "arent": "are not",
    "couldnt": "could not",
    "didnt": "did not",
    "doesnt": "does not",
    "dont": "do not",
    "hadnt": "had not",
    "hasnt": "has not",
    "havent": "have not",
    "hows": "how is",
    "im": "i am",
    "isnt": "is not",
    "ive": "i have",
    "shouldnt": "should not",
    "thats": "that is",
    "theyre": "they are",
    "wasnt": "was not",
    "weve": "we have",
    "werent": "were not",
    "whats": "what is",
    "wheres": "where is",
    "wouldnt": "would not",
    "youre": "you are",
    "youve": "you have",
}

CHAT_ALIASES = {
    "alot": ("a", "lot"),
    "gimme": ("give", "me"),
    "gonna": ("going", "to"),
    "gotta": ("got", "to"),
    "kinda": ("kind", "of"),
    "lemme": ("let", "me"),
    "pls": ("please",),
    "plz": ("please",),
    "thx": ("thanks",),
    "u": ("you",),
    "ur": ("your",),
    "wanna": ("want", "to"),
}

FILLERS = {
    "ah",
    "eh",
    "er",
    "erm",
    "hey",
    "hm",
    "hmm",
    "mm",
    "ok",
    "okay",
    "robot",
    "sieve",
    "uh",
    "um",
}

OBJECT_FILLERS = {
    "item",
    "items",
    "object",
    "objects",
    "stuff",
    "thing",
    "things",
}


def _expand_token(token: str) -> list[str]:
    if token in CHAT_ALIASES:
        return list(CHAT_ALIASES[token])
    if token in CONTRACTIONS:
        return CONTRACTIONS[token].split()
    if token.endswith("n't") and len(token) > 4:
        stem = token[:-3]
        special = {"ca": "can", "wo": "will", "sha": "shall"}
        return [special.get(stem, stem), "not"]
    if token.endswith("'s") and len(token) > 3:
        return [token[:-2]]
    for suffix, replacement in (
        ("'re", "are"),
        ("'ve", "have"),
        ("'ll", "will"),
        ("'d", "would"),
        ("'m", "am"),
    ):
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return [token[: -len(suffix)], replacement]
    return [token]


def expand_contractions(text: str) -> str:
    normalized = text.lower().replace("’", "'")
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", normalized)
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(_expand_token(token))
    return " ".join(expanded)


PHRASE_REWRITES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("give", "me", "a", "list", "of"), ("list",)),
    (("give", "me", "the", "list", "of"), ("list",)),
    (("a", "list", "of"), ("list",)),
    (("on", "the", "top", "of"), ("on",)),
    (("on", "top", "of"), ("on",)),
    (("inside", "of"), ("in",)),
    (("in", "the", "inside", "of"), ("in",)),
    (("out", "of"), ("from",)),
    (("off", "of"), ("from",)),
    (("away", "from"), ("from",)),
    (("adjacent", "to"), ("next", "to")),
    (("close", "to"), ("next", "to")),
    (("in", "front", "of"), ("next", "to")),
    (("pick", "up"), ("take",)),
    (("pick", "up", "the"), ("take", "the")),
    (("set", "down"), ("put",)),
    (("put", "down"), ("put",)),
    (("put", "away"), ("put",)),
    (("place", "down"), ("put",)),
    (("take", "hold", "of"), ("take",)),
    (("get", "hold", "of"), ("take",)),
    (("hand", "over"), ("give",)),
    (("give", "back"), ("give",)),
    # Common everyday phrasings for opening and closing a container.
    (("turn", "on"), ("open",)),
    (("switch", "on"), ("open",)),
    (("power", "on"), ("open",)),
    (("start", "up"), ("open",)),
    (("turn", "off"), ("close",)),
    (("switch", "off"), ("close",)),
    (("power", "off"), ("close",)),
    (("shut", "down"), ("close",)),
    # Directional idioms that are semantically redundant in this small world.
    (("over", "to"), ("to",)),
    (("over", "there"), ("there",)),
    (("over", "here"), ("here",)),
    (("down", "on"), ("on",)),
    (("down", "onto"), ("on",)),
    (("down", "in"), ("in",)),
    (("back", "on"), ("on",)),
    (("back", "onto"), ("on",)),
    (("back", "in"), ("in",)),
    (("back", "into"), ("in",)),
    (("back", "to"), ("to",)),
    (("back", "at"), ("at",)),
    (("in", "here"), ("here",)),
    (("in", "there"), ("there",)),
    (("inside", "here"), ("here",)),
    (("inside", "there"), ("there",)),
    (("in", "this", "room"), ("here",)),
    (("inside", "this", "room"), ("here",)),
    (("in", "the", "current", "room"), ("here",)),
    (("in", "my", "room"), ("here",)),
    (("in", "the", "same", "room"), ("here",)),
    (("all", "of", "the"), ("all", "the")),
    (("all", "of", "these"), ("all", "these")),
    (("all", "of", "those"), ("all", "those")),
    (("open", "up"), ("open",)),
    (("close", "up"), ("close",)),
    (("shut", "up"), ("close",)),
    (("go", "ahead", "and"), ()),
    (("go", "ahead",), ()),
    (("try", "to",), ()),
    (("be", "able", "to",), ()),
    (("located",), ()),
    (("situated",), ()),
    (("whereabouts",), ("where",)),
    (("colour",), ("color",)),
    (("hue",), ("color",)),
    (("tint",), ("color",)),
    (("within",), ("in",)),
    (("into",), ("in",)),
    (("onto",), ("on",)),
    (("atop",), ("on",)),
    (("upon",), ("on",)),
    (("beside",), ("next", "to")),
    (("near",), ("next", "to")),
    (("alongside",), ("next", "to")),
    (("toward",), ("to",)),
    (("towards",), ("to",)),
    (("off",), ("from",)),
)


_LIST_LEADERS = {"describe", "list", "name", "show"}
_LIST_FILLERS = {
    "all",
    "any",
    "each",
    "every",
    "everything",
    "item",
    "items",
    "object",
    "objects",
    "stuff",
    "the",
    "thing",
    "things",
}


def _replace_sequence(
    tokens: list[str], old: tuple[str, ...], new: tuple[str, ...]
) -> tuple[list[str], bool]:
    if not old:
        return tokens, False
    result: list[str] = []
    index = 0
    changed = False
    while index < len(tokens):
        if tuple(tokens[index : index + len(old)]) == old:
            result.extend(new)
            index += len(old)
            changed = True
        else:
            result.append(tokens[index])
            index += 1
    return result, changed


def rewrite_phrases(tokens: list[str]) -> list[str]:
    rewritten = _rewrite_separated_particles(list(tokens))
    changed = True
    while changed:
        changed = False
        for old, new in PHRASE_REWRITES:
            rewritten, replaced = _replace_sequence(rewritten, old, new)
            changed = changed or replaced
    return _rewrite_question_patterns(rewritten)


def _rewrite_separated_particles(tokens: list[str]) -> list[str]:
    """Normalize "turn/switch/power the X on|off" before generic rewrites."""

    if (
        len(tokens) >= 3
        and (
            (tokens[0] in {"power", "switch", "turn"} and tokens[-1] in {"from", "off", "on"})
            or (tokens[0] == "shut" and tokens[-1] == "down")
        )
    ):
        action = "open" if tokens[-1] == "on" else "close"
        middle = tokens[1:-1]
        if middle and middle[-1] == "back":
            middle = middle[:-1]
        return [action, *middle]
    return tokens


def _rewrite_question_patterns(tokens: list[str]) -> list[str]:
    if not tokens:
        return []

    # "Make the fridge open" is an imperative paraphrase, not a location.
    if len(tokens) >= 3 and tokens[0] == "make" and tokens[-1] in {
        "closed",
        "open",
        "shut",
    }:
        action = "open" if tokens[-1] == "open" else "close"
        return [action, *tokens[1:-1]]

    # Count questions often put "sit/stand" between the noun and location:
    # "how many cups are sitting on the table".
    if tokens[:2] == ["how", "many"]:
        for filler in ("sitting", "standing", "sit", "sits", "stand", "stands"):
            if filler not in tokens:
                continue
            index = tokens.index(filler)
            tokens.pop(index)
            if index > 0 and tokens[index - 1] in {"is", "are"}:
                tokens.pop(index - 1)
            break

    # Conversational inventory questions.
    if tokens[:4] in (["what", "have", "you", "got"], ["what", "has", "it", "got"]):
        tail = tokens[4:]
        return ["what", "is", *tail] if tail else ["what", "is", "everywhere"]
    if tokens[:4] in (["what", "do", "you", "have"], ["what", "does", "it", "have"]):
        tail = tokens[4:]
        if not tail:
            return ["what", "is", "everywhere"]
        if tail[0] in {
            "above",
            "at",
            "beside",
            "from",
            "in",
            "inside",
            "near",
            "next",
            "on",
            "under",
            "within",
        }:
            return ["what", "is", *tail]
        return ["what", "is", "in", *tail]

    # Inverted embedded questions: "the red cup is where".
    if len(tokens) >= 4 and tokens[-1] in {"where"} and tokens[-2] in {"is", "are"}:
        return [tokens[-1], tokens[-2], *tokens[:-2]]

    asked_for_else = tokens[:2] in (["what", "else"], ["anything", "else"])
    while tokens[:2] in (["what", "else"], ["anything", "else"]):
        tokens = ["what", *tokens[2:]]
    if asked_for_else and tokens == ["what"]:
        return ["what", "is", "everywhere"]

    # "Show me X" and "list X" are requests for a description or for the
    # contents of a place. Canonicalize them before the frame parser sees
    # them. "show me what ..." is already a question and is passed through.
    if tokens[:1] and tokens[0] in _LIST_LEADERS:
        tail = tokens[1:]
        if tail[:1] == ["me"]:
            tail = tail[1:]
        if tail[:1] in (["what"], ["where"], ["which"], ["how"], ["is"], ["are"]):
            return tail
        asked_for_kind = any(
            token in {"all", "any", "each", "every"} for token in tail
        )
        while tail and tail[0] in _LIST_FILLERS:
            tail = tail[1:]
        if not tail:
            return ["what", "is", "everywhere"]
        if tail[0] in {
            "at",
            "beside",
            "from",
            "in",
            "inside",
            "into",
            "near",
            "next",
            "on",
            "onto",
            "under",
            "within",
        }:
            return ["what", "is", *tail]
        if tail[0] in {
            "a",
            "an",
            "another",
            "its",
            "my",
            "some",
            "that",
            "the",
            "their",
            "these",
            "this",
            "those",
            "it",
            "them",
            "they",
            "your",
        }:
            return ["what", "is", *tail]
        if asked_for_kind or (
            tokens[0] in {"list", "name"}
            and tail[0] not in {"content", "contents"}
        ):
            return ["which", *tail]
        return ["what", "is", "the", *tail]

    # "Tell me about X" loses "tell me" as a polite prefix. A bare "about X"
    # is therefore a description request, not a location.
    if tokens[:1] == ["about"] and len(tokens) > 1:
        tail = tokens[1:]
        if tail[0] not in {
            "a",
            "an",
            "its",
            "my",
            "some",
            "that",
            "the",
            "their",
            "these",
            "this",
            "those",
            "your",
        }:
            tail = ["the", *tail]
        return ["what", "is", *tail]
    if tokens[:1] == ["more"] and tokens[1:2] == ["about"] and len(tokens) > 2:
        return ["what", "is", *tokens[2:]]
    if tokens[:4] == ["what", "do", "you", "know"] and tokens[4:5] == ["about"]:
        return ["what", "is", *tokens[5:]]
    if tokens[:4] == ["what", "can", "you", "tell"] and tokens[4:6] == ["me", "about"]:
        return ["what", "is", *tokens[6:]]
    if tokens[:5] == ["what", "can", "you", "tell", "me"] and tokens[5:6] == ["about"]:
        return ["what", "is", *tokens[6:]]

    # Embedded questions often keep the auxiliary at the end:
    # "what color the cup is" -> "what color is the cup".
    if (
        len(tokens) >= 4
        and tokens[0] == "what"
        and tokens[1] in {"color", "size"}
        and tokens[-1] in {"is", "are"}
        and tokens[2] not in {"is", "are"}
    ):
        return ["what", tokens[1], "is", *tokens[2:-1]]

    # Existential questions ask whether at least one matching thing exists.
    # The parser gives the resulting noun phrase an existence frame.
    if tokens[:3] in (["do", "you", "have"], ["do", "we", "have"]):
        tail = tokens[3:]
        auxiliary = "are" if tail[:1] in (["any"], ["some"]) else "is"
        return [auxiliary, "there", *tail]
    if tokens[:4] == ["have", "you", "got", "any"]:
        return ["are", "there", "any", *tokens[4:]]
    if tokens[:3] == ["is", "there", "anything"]:
        return ["is", "there", "a", "thing", *tokens[3:]]
    if tokens[:3] == ["is", "there", "something"]:
        return ["is", "there", "a", "thing", *tokens[3:]]
    if tokens[:4] == ["are", "there", "any", "things"]:
        return ["are", "there", "any", "things", *tokens[4:]]
    if tokens[:3] in (["what", "do", "you"], ["what", "can", "you"]):
        if tokens[3:4] == ["have"] or tokens[3:4] == ["see"]:
            tail = tokens[4:]
            if not tail:
                return ["what", "is", "everywhere"]
            return ["what", "is", *tail]
    if tokens[:2] == ["what", "is"] and tokens[2:3] == ["there"] and len(tokens) > 3:
        return ["what", "is", *tokens[3:]]

    # "How many cups does the table have?" uses possession to ask about
    # containment. Choose the natural relation from the named location.
    if tokens[:1] == ["does"] and any(
        token in {"contain", "have"} for token in tokens[2:]
    ):
        relation_word = "contain" if "contain" in tokens[2:] else "have"
        index = tokens.index(relation_word, 2)
        target = tokens[1:index]
        noun_part = tokens[index + 1 :]
        if target and noun_part:
            support_words = {"counter", "desk", "floor", "shelf", "table"}
            relation = "on" if target[-1] in support_words else "in"
            return ["is", "there", *noun_part, relation, *target]

    if tokens[:2] == ["how", "many"] and "does" in tokens and tokens[-1:] in (
        ["have"],
        ["contain"],
    ):
        index = tokens.index("does")
        noun_part = tokens[2:index]
        target = tokens[index + 1 : -1]
        if noun_part and target:
            support_words = {"counter", "desk", "floor", "shelf", "table"}
            relation = "on" if target[-1] in support_words else "in"
            return ["how", "many", *noun_part, "are", relation, *target]

    if len(tokens) >= 4 and tokens[0] == "where":
        if tokens[1] in {"can", "could", "do", "should", "would"}:
            if tokens[2] in {"i", "we", "you"} and tokens[3] in {
                "find",
                "get",
                "locate",
                "see",
            }:
                return ["where", "is", *tokens[4:]]
    if (
        tokens[:1] == ["where"]
        and len(tokens) >= 4
        and tokens[1] in {"can", "could", "may", "might", "must", "should", "would"}
        and tokens[-1:] == ["be"]
    ):
        return ["where", "is", *tokens[2:-1]]

    # A bare phrase such as "the color of X" can remain after a polite prefix.
    if tokens[:3] == ["the", "color", "of"]:
        return ["what", "color", "is", *tokens[3:]]
    if tokens[:3] == ["the", "size", "of"]:
        return ["what", "size", "is", *tokens[3:]]
    if tokens[:3] == ["the", "location", "of"]:
        return ["where", "is", *tokens[3:]]
    if tokens[:3] == ["the", "where", "of"]:
        return ["where", "is", *tokens[3:]]
    if tokens[:3] == ["the", "state", "of"]:
        return ["what", "is", *tokens[4:]]
    if tokens[:3] == ["the", "status", "of"]:
        return ["what", "is", *tokens[4:]]
    if tokens[:3] == ["the", "condition", "of"]:
        return ["what", "is", *tokens[4:]]
    if tokens[:5] == ["what", "is", "the", "state", "of"]:
        return ["what", "is", *tokens[5:]]
    if tokens[:5] == ["what", "is", "the", "status", "of"]:
        return ["what", "is", *tokens[5:]]
    if tokens[:5] == ["what", "is", "the", "condition", "of"]:
        return ["what", "is", *tokens[5:]]

    # "What things are on the table" asks the same thing as "what is on the
    # table". Generic object nouns are fillers in this frame.
    if tokens[:1] == ["what"]:
        tail = list(tokens[1:])
        while tail[:1] in (["all"], ["any"], ["some"], ["else"]):
            tail = tail[1:]
        if tokens[:2] in (["what", "all"], ["what", "else"]) and tail[:1] in (
            ["is"],
            ["are"],
        ):
            rest = tail[1:]
            if not rest:
                return ["what", "is", "everywhere"]
            return ["what", "is", *rest]
        if tokens[:2] == ["what", "are"] and tokens[2:3] in (
            ["all"],
            ["any"],
            ["some"],
            ["the"],
        ):
            extra = tokens[2:]
            while extra[:1] in (["all"], ["any"], ["some"], ["the"]):
                extra = extra[1:]
            if extra[:1] in (
                ["object"],
                ["objects"],
                ["thing"],
                ["things"],
                ["item"],
                ["items"],
                ["stuff"],
            ):
                rest = extra[1:]
                if not rest:
                    return ["what", "is", "everywhere"]
                return ["what", "is", *rest]
        if tail[:1] in (
            ["object"],
            ["objects"],
            ["thing"],
            ["things"],
            ["item"],
            ["items"],
            ["stuff"],
        ):
            tail = tail[1:]
            if tail[:1] in (["is"], ["are"]):
                tail = tail[1:]
            if tail[:1] == ["there"]:
                tail = tail[1:]
            return ["what", "is", *tail]
        if tail[:1] == ["is"] and tail[1:2] == ["there"]:
            if len(tail) == 2:
                return ["what", "is", "there"]
            return ["what", "is", *tail[2:]]
        if tail[:2] in (["is", "everything"], ["is", "all"]):
            return ["what", "is", "everywhere"]

    # "What cups are on the table?" is a which-question over a kind. The
    # plural was normalized to its singular noun while tokenizing, so the
    # auxiliary is the reliable marker of the original plural frame.
    if len(tokens) >= 5 and tokens[0] == "what":
        for index in range(2, len(tokens) - 1):
            if tokens[index] not in {"is", "are"}:
                continue
            if tokens[index + 1] not in {
                "at",
                "in",
                "next",
                "on",
                "under",
            }:
                continue
            noun_part = tokens[1:index]
            if not noun_part or noun_part[0] in OBJECT_FILLERS:
                break
            return ["which", *noun_part, tokens[index], *tokens[index + 1 :]]

    if tokens[:2] == ["what", "is"] and tokens[-1:] == ["like"]:
        return ["what", "is", *tokens[2:-1]]
    if tokens[:2] == ["what", "is"] and tokens[-1:] in (["color"], ["size"]):
        attribute = tokens[-1]
        subject = tokens[2:-1]
        if subject in (["its"], ["their"]):
            return [
                "what",
                attribute,
                "is",
                "it" if subject == ["its"] else "them",
            ]
        if subject:
            return ["what", attribute, "is", *subject]
    if tokens[:2] == ["what", "is"] and tokens[-1:] == ["size"]:
        subject = tokens[2:-1]
        if subject:
            return ["what", "size", "is", *subject]
    if tokens[:2] == ["what", "is"] and tokens[-1:] == ["location"]:
        subject = tokens[2:-1]
        if subject in (["its"], ["their"]):
            return ["where", "is", "it" if subject == ["its"] else "them"]
        if subject:
            return ["where", "is", *subject]
    if tokens[:2] == ["what", "are"] and tokens[-2:] == ["its", "contents"]:
        return ["what", "is", "in", "it"]
    if tokens[:2] == ["what", "size"] and tokens[2:3] == ["does"] and tokens[-1:] == ["have"]:
        return ["what", "size", "is", *tokens[3:-1]]
    if tokens[:2] == ["what", "does"] and tokens[-2:] == ["look", "like"] and len(tokens) > 4:
        return ["what", "is", *tokens[2:-2]]
    if tokens[:2] == ["how", "does"] and tokens[-1:] == ["look"] and len(tokens) > 4:
        return ["what", "is", *tokens[2:-1]]

    if tokens[:3] in (["how", "big", "is"], ["how", "large", "is"]):
        return ["what", "size", "is", *tokens[3:]]

    if (
        tokens[:1] == ["where"]
        and len(tokens) >= 3
        and tokens[-1] in {"is", "are"}
        and tokens[1] not in {"is", "are"}
    ):
        return ["where", tokens[-1], *tokens[1:-1]]
    if tokens[:2] == ["where", "all"]:
        return ["where", *tokens[2:]]
    if tokens[:1] == ["where"] and tokens[-1:] in (
        ["at"],
        ["found"],
        ["kept"],
        ["stored"],
    ):
        return ["where", *tokens[1:-1]]
    if tokens[:4] in (
        ["what", "is", "the", "location"],
        ["what", "is", "the", "whereabouts"],
    ):
        tail = tokens[4:]
        if tail and tail[0] == "of":
            tail = tail[1:]
        return ["where", "is", *tail]
    if tokens[:5] == ["what", "is", "the", "where", "of"]:
        return ["where", "is", *tokens[5:]]
    if tokens[:5] == ["tell", "me", "the", "location", "of"]:
        return ["where", "is", *tokens[5:]]
    if tokens[:2] in (
        ["what", "is"],
        ["what", "are"],
        ["what", "was"],
        ["what", "were"],
    ) and tokens[2:5] in (
        ["the", "contents", "of"],
        ["the", "content", "of"],
    ):
        return ["what", "is", "in", *tokens[5:]]
    if tokens[:3] in (
        ["contents", "of"],
        ["content", "of"],
        ["contents", "in"],
        ["content", "in"],
    ):
        return ["what", "is", "in", *tokens[3:]]
    if tokens[:5] in (
        ["what", "is", "the", "color", "of"],
        ["what", "are", "the", "color", "of"],
    ):
        return ["what", "color", "is", *tokens[5:]]
    if tokens[:5] in (
        ["what", "is", "the", "size", "of"],
        ["what", "are", "the", "size", "of"],
    ):
        return ["what", "size", "is", *tokens[5:]]
    if (
        tokens[:3] in (["give", "me", "the"], ["show", "me", "the"], ["tell", "me", "the"])
        and tokens[3:5] in (["contents", "of"], ["content", "of"])
    ):
        return ["what", "is", "in", *tokens[5:]]
    if tokens[:3] == ["what", "is", "contained"]:
        if tokens[3:4] == ["in"]:
            return ["what", "is", "in", *tokens[4:]]
    if tokens[:2] == ["what", "does"] and tokens[-1:] == ["contain"]:
        return ["what", "is", "in", *tokens[2:-1]]
    if len(tokens) >= 4 and tokens[0] == "does" and "contain" in tokens:
        index = tokens.index("contain")
        if index > 1 and index + 1 < len(tokens):
            return ["is", *tokens[index + 1 :], "in", *tokens[1:index]]
    if tokens[:2] == ["which", "color"]:
        return ["what", "color", *tokens[2:]]
    if tokens[:3] == ["what", "color", "does"] and tokens[-1:] == ["have"]:
        return ["what", "color", "is", *tokens[3:-1]]
    if tokens[:2] == ["what", "is"] and tokens[-1:] == ["color"]:
        subject = tokens[2:-1]
        if subject:
            return ["what", "color", "is", *subject]
    if tokens[:4] == ["in", "what", "room", "is"]:
        tail = tokens[4:]
        if tail[-1:] in (["at"], ["in"]):
            tail = tail[:-1]
        return ["where", "is", *tail]
    if tokens[:3] in (["what", "room", "is"], ["which", "room", "is"]):
        tail = tokens[3:]
        if tail[-1:] in (["at"], ["in"]):
            tail = tail[:-1]
        return ["where", "is", *tail]
    if tokens[:3] == ["do", "you", "see"] and any(
        token in {"on", "in", "at", "next"} for token in tokens[3:]
    ):
        return ["is", *tokens[3:]]
    if tokens[0] in {"list", "name"}:
        tail = [
            token
            for token in tokens[1:]
            if token not in {"all", "the", "every", "everything", "objects", "items", "things"}
        ]
        if "contents" in tail and "of" in tail:
            index = tail.index("of")
            return ["what", "is", "in", *tail[index + 1 :]]
        for index, token in enumerate(tail):
            if token in {"on", "in", "at", "next"}:
                return ["what", "is", *tail[index:]]
    return tokens


POLITE_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("would", "you", "be", "able", "to"),
    ("are", "you", "able", "to"),
    ("would", "you", "mind"),
    ("do", "you", "mind"),
    ("would", "you", "possibly"),
    ("could", "you", "possibly"),
    ("could", "you", "kindly"),
    ("would", "you", "kindly"),
    ("i", "would", "like", "you", "to"),
    ("i", "would", "like", "to"),
    ("i", "would", "like", "to", "know"),
    ("i", "want", "to", "know"),
    ("i", "need", "to", "know"),
    ("i", "am", "wondering"),
    ("i", "was", "wondering"),
    ("i", "want", "you", "to"),
    ("i", "want", "to"),
    ("i", "need", "you", "to"),
    ("i", "need", "to"),
    ("do", "you", "know"),
    ("do", "you", "happen", "to", "know"),
    ("if", "you", "could"),
    ("if", "you", "would"),
    ("i", "wonder", "if", "you", "could"),
    ("i", "wonder", "whether", "you", "could"),
    ("i", "wonder", "if"),
    ("i", "wonder", "whether"),
    ("i", "wonder"),
    ("could", "you"),
    ("would", "you"),
    ("can", "you"),
    ("will", "you"),
    ("help", "me", "to"),
    ("help", "me"),
    ("tell", "me"),
    ("let", "us"),
)
SINGLE_PREFIXES = {"just", "kindly", "please"}


def strip_polite_prefixes(tokens: list[str]) -> list[str]:
    remaining = list(tokens)
    changed = True
    while changed and remaining:
        changed = False
        while remaining and (remaining[0] in FILLERS or remaining[0] in SINGLE_PREFIXES):
            remaining.pop(0)
            changed = True
        for prefix in sorted(POLITE_PREFIXES, key=len, reverse=True):
            if tuple(remaining[: len(prefix)]) == prefix:
                del remaining[: len(prefix)]
                changed = True
                break
    return remaining
