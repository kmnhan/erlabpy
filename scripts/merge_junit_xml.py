from __future__ import annotations

import argparse
import copy
import pathlib
import xml.etree.ElementTree as ET

COUNT_ATTRS = ("tests", "failures", "errors", "skipped")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge multiple JUnit XML reports.")
    parser.add_argument("inputs", nargs="+", help="Input JUnit XML files.")
    parser.add_argument("--output", required=True, help="Output JUnit XML file.")
    return parser


def iter_suites(path: pathlib.Path) -> list[ET.Element]:
    root = ET.parse(path).getroot()  # noqa: S314 - local CI artifacts only
    if root.tag == "testsuite":
        return [root]
    if root.tag == "testsuites":
        return [child for child in root if child.tag == "testsuite"]
    raise ValueError(f"Unsupported JUnit XML root tag in {path}: {root.tag}")


def merge_reports(inputs: list[pathlib.Path], output: pathlib.Path) -> None:
    merged_root = ET.Element("testsuites")
    totals = dict.fromkeys(COUNT_ATTRS, 0)
    total_time = 0.0

    for input_path in inputs:
        for suite in iter_suites(input_path):
            merged_root.append(copy.deepcopy(suite))
            for attr in COUNT_ATTRS:
                totals[attr] += int(float(suite.attrib.get(attr, "0")))
            total_time += float(suite.attrib.get("time", "0"))

    for attr in COUNT_ATTRS:
        merged_root.set(attr, str(totals[attr]))
    merged_root.set("time", f"{total_time:.3f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged_root).write(output, encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    inputs = [pathlib.Path(path) for path in args.inputs]
    output = pathlib.Path(args.output)
    merge_reports(inputs, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
