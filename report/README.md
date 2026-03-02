# report

Typst source for the competition report.

## Build

Always build the PDF through the wrapper script:

```bash
./build.sh
```

This script prevents the Japanese text corruption that occurs when Typst is run
without the bundled font path. It does two things before compilation:

1. Verifies that Typst can see the bundled font families in `report/fonts/`
2. Compiles with `--font-path report/fonts`

To write to a different output path:

```bash
./build.sh ./report.typ ./out/report.pdf
```

## Why this exists

Running `typst compile report.typ report.pdf` directly can produce tofu glyphs
for Japanese text if the local environment does not already have compatible CJK
fonts installed. The repository already vendors the required fonts in
`report/fonts/`, so the wrapper script forces Typst to use them.
