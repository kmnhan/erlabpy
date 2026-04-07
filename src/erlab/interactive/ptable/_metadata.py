from __future__ import annotations

__all__ = [
    "CATEGORY_COLORS",
    "CATEGORY_REFERENCES",
    "CATEGORY_TITLES",
    "ELEMENT_CATEGORIES",
    "ELEMENT_POSITIONS",
    "GROUND_STATE_CONFIGURATIONS",
    "CategoryReference",
    "ReferenceSource",
    "configuration_to_html",
]

import html
import re
from dataclasses import dataclass

import xraydb


@dataclass(frozen=True)
class ReferenceSource:
    citation: str
    url: str
    citation_html: str | None = None


@dataclass(frozen=True)
class CategoryReference:
    title: str
    blurb: str
    references: tuple[ReferenceSource, ...]


def _book_reference(
    authors: str,
    title: str,
    publisher: str,
    city: str,
    year: int,
    url: str,
) -> ReferenceSource:
    citation = f"{authors} {title} ({publisher}, {city}, {year})."
    citation_html = (
        f"{html.escape(authors)} <i>{html.escape(title)}</i> "
        f"({html.escape(publisher)}, {html.escape(city)}, {year})."
    )
    return ReferenceSource(citation, url, citation_html)


def _journal_reference(
    authors: str,
    title: str,
    journal: str,
    volume: str,
    pages: str,
    year: int,
    url: str,
) -> ReferenceSource:
    citation = f"{authors} {title} {journal} {volume}, {pages} ({year})."
    citation_html = (
        f"{html.escape(authors)} {html.escape(title)} "
        f"<i>{html.escape(journal)}</i> <b>{html.escape(volume)}</b>, "
        f"{html.escape(pages)} ({year})."
    )
    return ReferenceSource(citation, url, citation_html)


def _online_reference(
    authors: str,
    title: str,
    work: str,
    year: int,
    url: str,
) -> ReferenceSource:
    citation = f"{authors} {title} {work} ({year})."
    citation_html = (
        f"{html.escape(authors)} {html.escape(title)} "
        f"<i>{html.escape(work)}</i> ({year})."
    )
    return ReferenceSource(citation, url, citation_html)


_RED_BOOK_URL = "https://publications.iupac.org/books/rbook/Red_Book_2005.pdf"
_TRANSITION_ELEMENT_URL = "https://doi.org/10.1351/goldbook.T06456"
_INERT_GAS_URL = "https://doi.org/10.1351/goldbook.I03027"
_METALLIC_CHARACTER_URL = "https://doi.org/10.1021/ed060p691"
_METAL_NONMETAL_REVIEW_URL = "https://doi.org/10.1098/rsta.2020.0213"
_METALLOID_SURVEY_URL = "https://doi.org/10.1021/ed3008457"
_SEMIMETALLICITY_URL = "https://doi.org/10.1021/ed078p1686"
_PO_ASTATINE_URL = "https://doi.org/10.1021/ed100308w"

_RED_BOOK_REFERENCE = _book_reference(
    "Connelly, N. G., Damhus, T., Hartshorn, R. M. & Hutton, A. T.",
    "Nomenclature of Inorganic Chemistry: IUPAC Recommendations 2005",
    "RSC Publishing",
    "Cambridge",
    2005,
    _RED_BOOK_URL,
)
_TRANSITION_ELEMENT_REFERENCE = _online_reference(
    "IUPAC.",
    "Transition element.",
    "Compendium of Chemical Terminology (the Gold Book)",
    2019,
    _TRANSITION_ELEMENT_URL,
)
_INERT_GAS_REFERENCE = _online_reference(
    "IUPAC.",
    "Inert gas.",
    "Compendium of Chemical Terminology (the Gold Book)",
    2019,
    _INERT_GAS_URL,
)
_METALLIC_CHARACTER_REFERENCE = _journal_reference(
    "Edwards, P. P. & Sienko, M. J.",
    "On the occurrence of metallic character in the periodic table of the elements.",
    "J. Chem. Educ.",
    "60",
    "691-696",
    1983,
    _METALLIC_CHARACTER_URL,
)
_METAL_NONMETAL_REVIEW_REFERENCE = _journal_reference(
    "Yao, B. et al.",
    "Metals and non-metals in the periodic table.",
    "Philos. Trans. R. Soc. A",
    "378",
    "20200213",
    2020,
    _METAL_NONMETAL_REVIEW_URL,
)
_GOLDSMITH_METALLOID_REFERENCE = _journal_reference(
    "Goldsmith, R. H.",
    "Metalloids.",
    "J. Chem. Educ.",
    "59",
    "526-527",
    1982,
    "https://doi.org/10.1021/ed059p526",
)
_SEMIMETALLICITY_REFERENCE = _journal_reference(
    "Hawkes, S. J.",
    "Semimetallicity?",
    "J. Chem. Educ.",
    "78",
    "1686-1687",
    2001,
    _SEMIMETALLICITY_URL,
)
_PO_ASTATINE_REFERENCE = _journal_reference(
    "Hawkes, S. J.",
    "Polonium and Astatine Are Not Semimetals.",
    "J. Chem. Educ.",
    "87",
    "783",
    2010,
    _PO_ASTATINE_URL,
)
_VERNON_METALLOID_REFERENCE = _journal_reference(
    "Vernon, R. E.",
    "Which Elements Are Metalloids?",
    "J. Chem. Educ.",
    "90",
    "1703-1707",
    2013,
    _METALLOID_SURVEY_URL,
)

CATEGORY_REFERENCES: dict[str, CategoryReference] = {
    "alkali_metal": CategoryReference(
        title="Alkali metals",
        blurb=(
            "This category covers the six Group 1 metals Li, Na, K, Rb, Cs, and Fr."
        ),
        references=(_RED_BOOK_REFERENCE,),
    ),
    "alkaline_earth_metal": CategoryReference(
        title="Alkaline earth metals",
        blurb=("This category covers the Group 2 metals Be, Mg, Ca, Sr, Ba, and Ra."),
        references=(_RED_BOOK_REFERENCE,),
    ),
    "transition_metal": CategoryReference(
        title="Transition metals",
        blurb=(
            "This app uses the common groups 3-12 display for transition metals. "
            "That includes Zn, Cd, and Hg, although stricter definitions do not "
            "always treat group 12 as transition elements."
        ),
        references=(_RED_BOOK_REFERENCE, _TRANSITION_ELEMENT_REFERENCE),
    ),
    "other_metal": CategoryReference(
        title="Other metals",
        blurb=(
            "This app groups Al, Ga, In, Sn, Tl, Pb, Bi, Po, Nh, Fl, Mc, and Lv as "
            "other metals. It is a display category for the remaining p-block "
            "metals, including polonium."
        ),
        references=(_METALLIC_CHARACTER_REFERENCE, _PO_ASTATINE_REFERENCE),
    ),
    "metalloid": CategoryReference(
        title="Metalloids",
        blurb=(
            "This app uses the common six-element metalloid set: B, Si, Ge, As, Sb, "
            "and Te. Borderline cases such as polonium are not included."
        ),
        references=(
            _GOLDSMITH_METALLOID_REFERENCE,
            _SEMIMETALLICITY_REFERENCE,
            _VERNON_METALLOID_REFERENCE,
        ),
    ),
    "nonmetal": CategoryReference(
        title="Non-metals",
        blurb=(
            "This app groups H, C, N, O, P, S, and Se as non-metals after "
            "separating the halogens and noble gases. It is a display label for "
            "the remaining nonmetallic elements."
        ),
        references=(
            _METALLIC_CHARACTER_REFERENCE,
            _METAL_NONMETAL_REVIEW_REFERENCE,
        ),
    ),
    "halogen": CategoryReference(
        title="Halogens",
        blurb=(
            "This category covers the Group 17 halogen family: F, Cl, Br, I, and At."
        ),
        references=(_RED_BOOK_REFERENCE,),
    ),
    "noble_gas": CategoryReference(
        title="Noble gases",
        blurb=(
            "This category covers the Group 18 noble gas family. "
            "Noble gases are unreactive toward most species under ordinary conditions."
        ),
        references=(_RED_BOOK_REFERENCE, _INERT_GAS_REFERENCE),
    ),
    "lanthanoid": CategoryReference(
        title="Lanthanoids",
        blurb=(
            "This category covers La-Lu. The IUPAC recommends lanthanoid over "
            "lanthanide as the name for this series, although both are widely used."
        ),
        references=(_RED_BOOK_REFERENCE,),
    ),
    "actinoid": CategoryReference(
        title="Actinoids",
        blurb=(
            "This category covers Ac-Lr. The IUPAC recommends actinoid over "
            "actinide as the name for this series, although both are widely used."
        ),
        references=(_RED_BOOK_REFERENCE,),
    ),
}

CATEGORY_TITLES: dict[str, str] = {
    **{key: reference.title for key, reference in CATEGORY_REFERENCES.items()},
    "unknown": "Unknown",
}


# Keep neighboring categories separated enough for the legend to stay legible
# without breaking familiar family groupings such as lanthanoids/actinoids.
CATEGORY_COLORS: dict[str, str] = {
    "alkali_metal": "#efb57a",
    "alkaline_earth_metal": "#efcf74",
    "transition_metal": "#88b168",
    "other_metal": "#8eaedc",
    "metalloid": "#70c0c9",
    "nonmetal": "#efb9ab",
    "halogen": "#edf2a5",
    "noble_gas": "#b6e7f4",
    "lanthanoid": "#cdc7eb",
    "actinoid": "#aca6dc",
    "unknown": "#d6dde4",
}


_MAIN_TABLE_LAYOUT: dict[int, dict[int, str]] = {
    1: {1: "H", 18: "He"},
    2: {1: "Li", 2: "Be", 13: "B", 14: "C", 15: "N", 16: "O", 17: "F", 18: "Ne"},
    3: {1: "Na", 2: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar"},
    4: {
        1: "K",
        2: "Ca",
        3: "Sc",
        4: "Ti",
        5: "V",
        6: "Cr",
        7: "Mn",
        8: "Fe",
        9: "Co",
        10: "Ni",
        11: "Cu",
        12: "Zn",
        13: "Ga",
        14: "Ge",
        15: "As",
        16: "Se",
        17: "Br",
        18: "Kr",
    },
    5: {
        1: "Rb",
        2: "Sr",
        3: "Y",
        4: "Zr",
        5: "Nb",
        6: "Mo",
        7: "Tc",
        8: "Ru",
        9: "Rh",
        10: "Pd",
        11: "Ag",
        12: "Cd",
        13: "In",
        14: "Sn",
        15: "Sb",
        16: "Te",
        17: "I",
        18: "Xe",
    },
    6: {
        1: "Cs",
        2: "Ba",
        3: "La",
        4: "Hf",
        5: "Ta",
        6: "W",
        7: "Re",
        8: "Os",
        9: "Ir",
        10: "Pt",
        11: "Au",
        12: "Hg",
        13: "Tl",
        14: "Pb",
        15: "Bi",
        16: "Po",
        17: "At",
        18: "Rn",
    },
    7: {
        1: "Fr",
        2: "Ra",
        3: "Ac",
        4: "Rf",
        5: "Db",
        6: "Sg",
        7: "Bh",
        8: "Hs",
        9: "Mt",
        10: "Ds",
        11: "Rg",
        12: "Cn",
        13: "Nh",
        14: "Fl",
        15: "Mc",
        16: "Lv",
        17: "Ts",
        18: "Og",
    },
}


_SERIES_LAYOUT: dict[int, dict[int, str]] = {
    9: {
        4: "Ce",
        5: "Pr",
        6: "Nd",
        7: "Pm",
        8: "Sm",
        9: "Eu",
        10: "Gd",
        11: "Tb",
        12: "Dy",
        13: "Ho",
        14: "Er",
        15: "Tm",
        16: "Yb",
        17: "Lu",
    },
    10: {
        4: "Th",
        5: "Pa",
        6: "U",
        7: "Np",
        8: "Pu",
        9: "Am",
        10: "Cm",
        11: "Bk",
        12: "Cf",
        13: "Es",
        14: "Fm",
        15: "Md",
        16: "No",
        17: "Lr",
    },
}


_POSITION_BY_SYMBOL = {
    symbol: (row, column)
    for layout in (_MAIN_TABLE_LAYOUT, _SERIES_LAYOUT)
    for row, columns in layout.items()
    for column, symbol in columns.items()
}


ELEMENT_POSITIONS: dict[int, tuple[int, int]] = {
    atomic_number: _POSITION_BY_SYMBOL[xraydb.atomic_symbol(atomic_number)]
    for atomic_number in range(1, 119)
}


_NONMETALS = {"H", "C", "N", "O", "P", "S", "Se"}
_METALLOIDS = {"B", "Si", "Ge", "As", "Sb", "Te"}
_OTHER_METALS = {
    "Al",
    "Ga",
    "In",
    "Sn",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
}


def _category_for(atomic_number: int, symbol: str, group: int) -> str:
    if 57 <= atomic_number <= 71:
        return "lanthanoid"
    if 89 <= atomic_number <= 103:
        return "actinoid"
    if group == 1 and symbol != "H":
        return "alkali_metal"
    if group == 2:
        return "alkaline_earth_metal"
    if group == 17:
        return "halogen"
    if group == 18:
        return "noble_gas"
    if symbol in _NONMETALS:
        return "nonmetal"
    if symbol in _METALLOIDS:
        return "metalloid"
    if symbol in _OTHER_METALS:
        return "other_metal"
    if 3 <= group <= 12:
        return "transition_metal"
    return "unknown"


ELEMENT_CATEGORIES: dict[int, str] = {
    atomic_number: _category_for(
        atomic_number,
        xraydb.atomic_symbol(atomic_number),
        ELEMENT_POSITIONS[atomic_number][1],
    )
    for atomic_number in range(1, 119)
}


GROUND_STATE_CONFIGURATIONS: dict[str, str] = {
    "H": "1s^1",
    "He": "1s^2",
    "Li": "[He] 2s^1",
    "Be": "[He] 2s^2",
    "B": "[He] 2s^2 2p^1",
    "C": "[He] 2s^2 2p^2",
    "N": "[He] 2s^2 2p^3",
    "O": "[He] 2s^2 2p^4",
    "F": "[He] 2s^2 2p^5",
    "Ne": "[He] 2s^2 2p^6",
    "Na": "[Ne] 3s^1",
    "Mg": "[Ne] 3s^2",
    "Al": "[Ne] 3s^2 3p^1",
    "Si": "[Ne] 3s^2 3p^2",
    "P": "[Ne] 3s^2 3p^3",
    "S": "[Ne] 3s^2 3p^4",
    "Cl": "[Ne] 3s^2 3p^5",
    "Ar": "[Ne] 3s^2 3p^6",
    "K": "[Ar] 4s^1",
    "Ca": "[Ar] 4s^2",
    "Sc": "[Ar] 3d^1 4s^2",
    "Ti": "[Ar] 3d^2 4s^2",
    "V": "[Ar] 3d^3 4s^2",
    "Cr": "[Ar] 3d^5 4s^1",
    "Mn": "[Ar] 3d^5 4s^2",
    "Fe": "[Ar] 3d^6 4s^2",
    "Co": "[Ar] 3d^7 4s^2",
    "Ni": "[Ar] 3d^8 4s^2",
    "Cu": "[Ar] 3d^10 4s^1",
    "Zn": "[Ar] 3d^10 4s^2",
    "Ga": "[Ar] 3d^10 4s^2 4p^1",
    "Ge": "[Ar] 3d^10 4s^2 4p^2",
    "As": "[Ar] 3d^10 4s^2 4p^3",
    "Se": "[Ar] 3d^10 4s^2 4p^4",
    "Br": "[Ar] 3d^10 4s^2 4p^5",
    "Kr": "[Ar] 3d^10 4s^2 4p^6",
    "Rb": "[Kr] 5s^1",
    "Sr": "[Kr] 5s^2",
    "Y": "[Kr] 4d^1 5s^2",
    "Zr": "[Kr] 4d^2 5s^2",
    "Nb": "[Kr] 4d^4 5s^1",
    "Mo": "[Kr] 4d^5 5s^1",
    "Tc": "[Kr] 4d^5 5s^2",
    "Ru": "[Kr] 4d^7 5s^1",
    "Rh": "[Kr] 4d^8 5s^1",
    "Pd": "[Kr] 4d^10",
    "Ag": "[Kr] 4d^10 5s^1",
    "Cd": "[Kr] 4d^10 5s^2",
    "In": "[Kr] 4d^10 5s^2 5p^1",
    "Sn": "[Kr] 4d^10 5s^2 5p^2",
    "Sb": "[Kr] 4d^10 5s^2 5p^3",
    "Te": "[Kr] 4d^10 5s^2 5p^4",
    "I": "[Kr] 4d^10 5s^2 5p^5",
    "Xe": "[Kr] 4d^10 5s^2 5p^6",
    "Cs": "[Xe] 6s^1",
    "Ba": "[Xe] 6s^2",
    "La": "[Xe] 5d^1 6s^2",
    "Ce": "[Xe] 4f^1 5d^1 6s^2",
    "Pr": "[Xe] 4f^3 6s^2",
    "Nd": "[Xe] 4f^4 6s^2",
    "Pm": "[Xe] 4f^5 6s^2",
    "Sm": "[Xe] 4f^6 6s^2",
    "Eu": "[Xe] 4f^7 6s^2",
    "Gd": "[Xe] 4f^7 5d^1 6s^2",
    "Tb": "[Xe] 4f^9 6s^2",
    "Dy": "[Xe] 4f^10 6s^2",
    "Ho": "[Xe] 4f^11 6s^2",
    "Er": "[Xe] 4f^12 6s^2",
    "Tm": "[Xe] 4f^13 6s^2",
    "Yb": "[Xe] 4f^14 6s^2",
    "Lu": "[Xe] 4f^14 5d^1 6s^2",
    "Hf": "[Xe] 4f^14 5d^2 6s^2",
    "Ta": "[Xe] 4f^14 5d^3 6s^2",
    "W": "[Xe] 4f^14 5d^4 6s^2",
    "Re": "[Xe] 4f^14 5d^5 6s^2",
    "Os": "[Xe] 4f^14 5d^6 6s^2",
    "Ir": "[Xe] 4f^14 5d^7 6s^2",
    "Pt": "[Xe] 4f^14 5d^9 6s^1",
    "Au": "[Xe] 4f^14 5d^10 6s^1",
    "Hg": "[Xe] 4f^14 5d^10 6s^2",
    "Tl": "[Hg] 6p^1",
    "Pb": "[Hg] 6p^2",
    "Bi": "[Hg] 6p^3",
    "Po": "[Hg] 6p^4",
    "At": "[Hg] 6p^5",
    "Rn": "[Hg] 6p^6",
    "Fr": "[Rn] 7s^1",
    "Ra": "[Rn] 7s^2",
    "Ac": "[Rn] 6d^1 7s^2",
    "Th": "[Rn] 6d^2 7s^2",
    "Pa": "[Rn] 5f^2 6d^1 7s^2",
    "U": "[Rn] 5f^3 6d^1 7s^2",
    "Np": "[Rn] 5f^4 6d^1 7s^2",
    "Pu": "[Rn] 5f^6 7s^2",
    "Am": "[Rn] 5f^7 7s^2",
    "Cm": "[Rn] 5f^7 6d^1 7s^2",
    "Bk": "[Rn] 5f^9 7s^2",
    "Cf": "[Rn] 5f^10 7s^2",
    "Es": "[Rn] 5f^11 7s^2",
    "Fm": "[Rn] 5f^12 7s^2",
    "Md": "[Rn] 5f^13 7s^2",
    "No": "[Rn] 5f^14 7s^2",
    "Lr": "[Rn] 5f^14 7s^2 7p^1",
    "Rf": "[Rn] 5f^14 6d^2 7s^2",
    "Db": "[Rn] 5f^14 6d^3 7s^2",
    "Sg": "[Rn] 5f^14 6d^4 7s^2",
    "Bh": "[Rn] 5f^14 6d^5 7s^2",
    "Hs": "[Rn] 5f^14 6d^6 7s^2",
}


_CONFIG_PATTERN = re.compile(r"(\d+[spdfg])\^(\d+)")


def configuration_to_html(configuration: str) -> str:
    if not configuration:
        return ""
    escaped = html.escape(configuration)
    return _CONFIG_PATTERN.sub(r"\1<sup>\2</sup>", escaped).replace(" ", "")
