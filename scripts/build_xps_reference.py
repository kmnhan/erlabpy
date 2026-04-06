"""Build the bundled XPS cross-section reference archive from Elettra downloads.

The Elettra WebCrossSections interface republishes the Yeh and Lindau atomic
photoionization calculations as dense tabulated text downloads. This script scrapes the
published element and subshell tables and converts them into the compact `.npz` archive
bundled with ERLabPy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import re
import ssl
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import numpy as np
import xraydb

BASE_URL = "https://vuo.elettra.eu/services/elements/"
LANDING_PAGE = "WebElements.html"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "erlab"
    / "analysis"
    / "_data"
    / "yeh_lindau_1985_pics.npz"
)
MAX_WORKERS = 12
ELEMENT_LINK_RE = re.compile(r"mnu_elem\.cgi\?ELEMENT=([A-Za-z]+)")
CHECKBOX_RE = re.compile(
    r'name=["\']?C\d+["\']?\s+value=["\']?([0-9]+[spdfghi])["\']?',
    re.IGNORECASE,
)
SUBSHELL_RE = re.compile(r"^(\d+)([spdfghi])$", re.IGNORECASE)
LEGACY_SYMBOL_MAP: dict[str, str] = {
    "Uun": "Ds",
    "Uuu": "Rg",
    "Uub": "Cn",
}

FetchText = Callable[[str], str]
FetchJob = tuple[str, str]


class EmptySubshellTableError(ValueError):
    """Raised when a site subshell table is empty."""


def subshell_sort_key(subshell: str) -> tuple[int, int, str]:
    """Sort subshells in a physically sensible order."""
    match = SUBSHELL_RE.fullmatch(str(subshell).strip())
    if match is None:
        return (999, 999, str(subshell))

    n = int(match.group(1))
    lchar = match.group(2).lower()
    l_order = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}
    return (n, l_order[lchar], str(subshell))


def unique_symbols(values: Sequence[str]) -> tuple[str, ...]:
    """Return values with duplicates removed while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def parse_element_symbols(html: str) -> tuple[str, ...]:
    """Extract element symbols from the landing page."""
    return unique_symbols(ELEMENT_LINK_RE.findall(html))


def parse_element_subshells(html: str) -> tuple[str, ...]:
    """Extract subshell labels from an element page."""
    subshells = CHECKBOX_RE.findall(html)
    return unique_symbols(tuple(label.lower() for label in subshells))


def parse_subshell_table(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse one subshell table and return photon energy and sigma arrays."""
    if not text.strip():
        raise EmptySubshellTableError("Subshell table is empty")

    table = np.loadtxt(io.StringIO(text), ndmin=2)
    if table.size == 0:
        raise EmptySubshellTableError("Subshell table is empty")
    if table.shape[1] < 2:
        raise ValueError("Subshell table must contain at least two columns")
    return (
        table[:, 0].astype(np.float64, copy=False),
        table[:, 1].astype(np.float64, copy=False),
    )


def canonical_symbol(site_symbol: str) -> str:
    """Map a site symbol to the canonical xraydb symbol."""
    mapped = LEGACY_SYMBOL_MAP.get(site_symbol, site_symbol)
    return xraydb.atomic_symbol(mapped)


def build_reference_archive(
    fetch_text: FetchText,
    *,
    base_url: str = BASE_URL,
    progress: Callable[[str], None] | None = None,
) -> dict[str, np.ndarray]:
    """Build the bundled archive contents from the WebCrossSections site."""
    arrays: dict[str, np.ndarray] = {}
    z_by_symbol: dict[str, int] = {}

    landing_html = fetch_text(urljoin(base_url, LANDING_PAGE))
    site_symbols = parse_element_symbols(landing_html)

    element_pages: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {
            executor.submit(
                fetch_text, urljoin(base_url, f"mnu_elem.cgi?ELEMENT={site_symbol}")
            ): site_symbol
            for site_symbol in site_symbols
        }
        for future in concurrent.futures.as_completed(future_to_symbol):
            site_symbol = future_to_symbol[future]
            element_pages[site_symbol] = future.result()

    fetch_jobs: dict[FetchJob, str] = {}
    pending_subshells: dict[str, int] = {}
    successful_subshells: dict[str, list[str]] = {}
    for site_symbol in site_symbols:
        symbol = canonical_symbol(site_symbol)
        element_html = element_pages[site_symbol]
        subshells = parse_element_subshells(element_html)
        if not subshells:
            continue

        ordered_subshells = tuple(sorted(subshells, key=subshell_sort_key))
        z_by_symbol[symbol] = xraydb.atomic_number(symbol)
        pending_subshells[symbol] = len(ordered_subshells) + 1
        successful_subshells[symbol] = []
        fetch_jobs[(symbol, "total")] = urljoin(
            base_url, f"data/{site_symbol.lower()}.txt"
        )

        for subshell in ordered_subshells:
            fetch_jobs[(symbol, subshell)] = urljoin(
                base_url, f"data/{site_symbol.lower()}{subshell}.txt"
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {
            executor.submit(fetch_text, url): job for job, url in fetch_jobs.items()
        }
        for future in concurrent.futures.as_completed(future_to_job):
            symbol, label = future_to_job[future]
            try:
                hv, sigma = parse_subshell_table(future.result())
            except EmptySubshellTableError:
                pass
            else:
                base = f"{symbol}__{label}"
                arrays[f"{base}__hv"] = hv.astype(np.float32, copy=False)
                arrays[f"{base}__sigma"] = sigma.astype(np.float32, copy=False)
                if label != "total":
                    successful_subshells[symbol].append(label)
            pending_subshells[symbol] -= 1
            if pending_subshells[symbol] == 0 and progress is not None:
                atomic_number = z_by_symbol[symbol]
                n_subshells = len(successful_subshells[symbol])
                progress(
                    f"Fetched Z={atomic_number} {symbol} ({n_subshells} subshells)"
                )

    empty_symbols = [
        symbol for symbol, subshells in successful_subshells.items() if not subshells
    ]
    for symbol in empty_symbols:
        z_by_symbol.pop(symbol, None)

    for symbol, subshells in successful_subshells.items():
        if not subshells:
            continue
        arrays[f"{symbol}__subshells"] = np.array(
            sorted(subshells, key=subshell_sort_key), dtype="U8"
        )

    if not z_by_symbol:
        raise ValueError("No populated element pages were found")

    items = sorted(z_by_symbol.items(), key=lambda item: item[1])
    arrays["_symbols"] = np.array([symbol for symbol, _ in items], dtype="U3")
    arrays["_Z"] = np.array([z for _, z in items], dtype=np.int16)
    return arrays


def write_reference_archive(
    arrays: dict[str, np.ndarray], out_path: str | Path
) -> None:
    """Write the archive contents to disk."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def build_ssl_context() -> ssl.SSLContext:
    """Create the TLS context used for the Elettra website.

    The server currently presents a certificate chain that is not verified by the
    default Python trust store, so the workaround is scoped to this fetch path.
    """
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def make_fetcher(
    *,
    timeout: float = 30.0,
    ssl_context: ssl.SSLContext | None = None,
) -> FetchText:
    """Create a text fetcher for the WebCrossSections site."""

    def fetch_text(url: str) -> str:
        if urlsplit(url).scheme != "https":
            raise ValueError(f"Unsupported URL scheme for {url}")
        request = urllib.request.Request(  # noqa: S310
            url,
            headers={"User-Agent": "ERLabPy XPS reference builder"},
        )
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=timeout, context=ssl_context
        ) as response:
            charset = response.headers.get_content_charset() or "latin-1"
            return response.read().decode(charset, errors="replace")

    return fetch_text


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the bundled ERLabPy XPS cross-section archive from the Elettra "
            "WebCrossSections downloads."
        )
    )
    parser.add_argument(
        "out_path",
        nargs="?",
        default=DEFAULT_OUTPUT,
        type=Path,
        help="Output path for the generated .npz archive.",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="Base URL of the WebCrossSections site.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the archive builder."""
    args = parse_args(argv)
    fetch_text = make_fetcher(ssl_context=build_ssl_context())

    def report_progress(message: str) -> None:
        print(message, flush=True)

    arrays = build_reference_archive(
        fetch_text, base_url=args.base_url, progress=report_progress
    )
    write_reference_archive(arrays, args.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
