# Plotting styles and colormaps

(reference-plotting-styles)=

## Bundled style sheets

Importing {mod}`erlab.plotting` registers the following Matplotlib style sheets. Styles
that select a font require that font on the system.

| Style sheet | Description |
| --- | --- |
| `erlab.general` | General-purpose ERLab figure defaults |
| `erlab.nature` | Thin lines and tick sizes that resemble Springer Nature journal figures |
| `erlab.arial` | Arial text and math fonts |
| `erlab.times` | Times New Roman text and the STIX math font |
| `erlab.helvetica` | Helvetica text and math fonts |
| `erlab.stixsans-fallback` | STIX Sans as a fallback math font for missing glyphs |

Style sheets compose from left to right. A later style can replace settings from an
earlier style. Use {ref}`how-to-plotting-figure-styles` to apply them in Python or Figure
Composer.

## External colormap libraries

The following optional libraries provide Matplotlib-compatible colormaps:

- [CMasher](https://github.com/1313e/CMasher)
- [cmocean](https://github.com/matplotlib/cmocean)
- [colorcet](https://github.com/holoviz/colorcet)
- [cmcrameri](https://github.com/callumrollo/cmcrameri)
