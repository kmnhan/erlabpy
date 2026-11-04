# Plotting styles and colormaps

(reference-plotting-styles)=

## Bundled style sheets

Importing {mod}`erlab.plotting` registers the following Matplotlib style sheets. Styles
that select a font require that font on the system.

| Style sheet | Description |
| --- | --- |
| `erlab.general` | General-purpose ERLab figure defaults |
| `erlab.nature` | For Springer Nature journal figures |
| `erlab.physrev` | For Physical Review journal figures |
| `erlab.arial` | Arial text and math fonts |
| `erlab.times` | Times New Roman text and the STIX math font |
| `erlab.helvetica` | Helvetica text and math fonts |
| `erlab.stixsans-fallback` | STIX Sans as a fallback math font for missing glyphs |

Use {ref}`how-to-plotting-figure-styles` to apply them in Python or Figure Composer.
Journal-specific styles also include some defaults biased on personal preference. Check
the journal's author guidelines before submitting a manuscript to ensure that the figure
style is acceptable.

## External colormap libraries

The following optional libraries provide additional Matplotlib-compatible colormaps:

- [CMasher](https://github.com/1313e/CMasher)
- [cmocean](https://github.com/matplotlib/cmocean)
- [colorcet](https://github.com/holoviz/colorcet)
- [cmcrameri](https://github.com/callumrollo/cmcrameri)
