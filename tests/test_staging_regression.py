"""The staging regression set: one table, every kind of movie.

Each row is a PAIR SITUATION taken from a real order, with the staging the
owner expects. The decision is code (`unstageable_reason`), so this runs
free and offline before every deploy.

It exists because fixes used to be made while looking at one movie and
checked against nothing else: a rule that was right for a family trip
(separate look-alikes by relative age) broke a growing-up movie, and so on.
No movie is one "type" - a growing-up movie, a family movie and a
father-and-son movie are the same handful of situations in different
proportions - so a change to the gate must keep EVERY row here true.
When the owner corrects a staging by hand, add that pair as a row.
"""
from __future__ import annotations

import pytest

from ai_video_maker.clients.openai_client import (
    camera_shift_prompt,
    is_camera_transition,
    unstageable_reason,
)

BOY = "the brown-haired boy"
BROTHER = "the blue-eyed boy"
DAD = "the dark-haired man with stubble"
MUM = "the woman with tied-back dark hair"
GIRL = "the girl with a high ponytail"
TEEN = "the long-haired teenage girl"
FAMILY = [DAD, BOY, TEEN, BROTHER, MUM]
BIG = [f"relative {i}" for i in range(10)]

PEOPLE, CAMERA = "people", "camera"

# (name, start roster, end roster, start heads, end heads, same place, expected)
CASES = [
    # --- a child growing up (am-101126-unaq) --------------------------------
    ("unaq 12: one boy, toboggan -> ski slope with tiny skiers behind",
     [BOY], [BOY], 1, 1, False, PEOPLE),
    ("unaq 15: one boy, mountain -> Pisa with tourists behind a fence",
     [BOY], [BOY], 1, 1, False, PEOPLE),
    ("unaq 18: the same boy at two ages, different places",
     [BOY], [BOY], 1, 1, False, PEOPLE),
    ("unaq 19: boy alone -> boy and brother, new place (owner staged an entrance)",
     [BOY], [BOY, BROTHER], 1, 2, False, PEOPLE),
    ("unaq 23: boy and teen in a car -> boy alone in a cafe (owner kept people)",
     [BOY, TEEN], [BOY], 2, 1, False, PEOPLE),
    ("unaq 8: two children on a couch -> two OTHER children on a floor "
     "(owner forced the camera)",
     ["the toddler with a dummy", GIRL], [BOY, "the baby in blue"], 2, 2, False,
     CAMERA),
    ("unaq 28: family photo -> another family photo, group rearranged",
     FAMILY, FAMILY[::-1], 5, 5, False, CAMERA),
    ("unaq 29: family of five -> two of them",
     FAMILY, [MUM, BOY], 5, 2, False, CAMERA),
    # --- a big family trip (am-130826-pcfd) ---------------------------------
    ("pcfd 08_to_09: ten at dinner -> six in a pool, eight people change",
     BIG, BIG[:4] + ["uncle", "aunt"], 10, 6, False, CAMERA),
    ("pcfd 0a_to_0b: six on a walkway -> five in a hallway, one leaves",
     BIG[:6], BIG[:5], 6, 5, False, CAMERA),
    ("pcfd 12_to_13: ten aboard but two tagged -> one of them in a pool",
     [BIG[0], BIG[1]], [BIG[0]], 10, 1, False, CAMERA),
    ("pcfd: the same big group holding a pose in one place",
     BIG, list(BIG), 10, 10, True, PEOPLE),
    ("pcfd 0o_to_01: one couple out, another couple in (four movers)",
     ["young woman with dark hair bun", "man with short brown hair"],
     ["older man with short gray hair", "woman with long curly hair"],
     2, 2, True, CAMERA),
    # --- two people, precise staging (father and son) -----------------------
    ("father and son stay, same place",
     [DAD, BOY], [DAD, BOY], 2, 2, True, PEOPLE),
    ("father and son, new place, nobody changes",
     [DAD, BOY], [DAD, BOY], 2, 2, False, PEOPLE),
    ("father and son trade sides (restaged in code, still people)",
     [DAD, BOY], [BOY, DAD], 2, 2, True, PEOPLE),
    ("son alone -> father joins him, same place",
     [BOY], [DAD, BOY], 1, 2, True, PEOPLE),
    ("father leaves, son stays, new place",
     [DAD, BOY], [BOY], 2, 1, False, PEOPLE),
    ("father alone -> son alone, same room (exit and entrance)",
     [DAD], [BOY], 1, 1, True, PEOPLE),
    ("father alone -> son alone, a different place: nothing carries over",
     [DAD], [BOY], 1, 1, False, CAMERA),
    # --- what must never decide -----------------------------------------------
    ("place unknown (planned before the field existed): old behaviour",
     [DAD], [BOY], 1, 1, None, PEOPLE),
    ("rosters that cannot be matched never buy a camera",
     ["the man", "the woman"], ["the tall man", "the short man"], 2, 2, False,
     PEOPLE),
]


@pytest.mark.parametrize(
    "name,start,end,start_heads,end_heads,same_place,expected",
    CASES, ids=[c[0] for c in CASES],
)
def test_the_staging_decision(name, start, end, start_heads, end_heads,
                              same_place, expected):
    reason = unstageable_reason(
        start, end, start_heads, end_heads, same_place=same_place)
    assert (CAMERA if reason else PEOPLE) == expected, (name, reason)


def test_every_camera_row_gets_a_real_camera_prompt():
    for name, start, end, *_ in (c for c in CASES if c[-1] == CAMERA):
        assert is_camera_transition(camera_shift_prompt(start, end)), name


def test_all_three_kinds_of_movie_are_represented():
    names = " ".join(c[0] for c in CASES)
    for kind in ("unaq", "pcfd", "father and son"):
        assert kind in names
    assert {c[-1] for c in CASES} == {PEOPLE, CAMERA}
