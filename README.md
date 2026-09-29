# TG-FAHFA HDCS Annotation Software

A desktop application for **structural annotation of TG-FAHFA lipids using EAciD-MS/MS spectra**.

The software integrates TG-FAHFA candidate generation, theoretical fragmentation prediction, experimental spectrum matching, hierarchical diagnostic-ion evaluation, HDCS scoring, positional-isomer assessment, and mixture analysis in a single workflow.

## Features

* Automated generation of TG-FAHFA structural candidates
* EAciD-MS/MS theoretical spectrum prediction
* Experimental/theoretical spectrum matching
* Hierarchical diagnostic-ion evaluation (L1–L4)
* WDIC, NEP, and HDCS scoring
* FAHFA *sn*-position assessment
* Internal FAHFA ester-bond position assessment
* Positional-isomer and mixture detection
* NNLS-assisted mixture analysis
* Interactive experimental/predicted butterfly spectrum
* Excel export of annotation results and diagnostic-ion evidence

## Workflow

```text 
Experimental EAciD-MS/MS spectrum
                │
                ▼
       Precursor assignment
                │
                ▼
      TG-FAHFA candidate
          generation
                │
                ▼
      Theoretical spectrum
          prediction
                │
                ▼
 Experimental/theoretical
       spectrum matching
                │
                ▼
 Hierarchical HDCS evidence
       L1 → L2 → L3 → L4
                │
                ▼
    HDCS scoring and
   positional assessment
                │
                ▼
 Candidate ranking / mixture
      analysis / export
```

## Hierarchical Structural Evidence

Diagnostic ions are organized into four evidence levels:

| Level  | Structural information                                   |
| ------ | -------------------------------------------------------- |
| **L1** | General TG-FAHFA fragmentation and neutral-loss evidence |
| **L2** | FAHFA and FA3 chain-related evidence                     |
| **L3** | FAHFA *sn*-position evidence                             |
| **L4** | FAHFA internal ester-bond position evidence              |

The scoring framework combines diagnostic-ion coverage with missing-evidence penalties and position-specific L4 support. Contradictory evidence is evaluated separately during positional-isomer and mixture assessment. Position-specific assignments are retained as unresolved when sufficient L4 evidence is unavailable.

## Input

The program accepts `.csv` peak-list files.

At minimum, the input must contain:

```text 
m/z
intensity
```

A built-in preview window allows manual mapping of source columns to `m/z` and `intensity`.

Example:

```csv 
m/z,intensity
255.2321,1250
283.2635,8320
497.3618,4280
925.7712,15700
1021.8408,85600
```

The highest-intensity valid peak is used as the precursor ion.

A demo CSV file is available from the **Releases** page for testing the workflow.

## Installation

Download the latest packaged application from the **Releases** page:

- **Windows:** download the `.exe` file and run it directly.
- **macOS:** download the `.zip` file, extract it, and open the `.app` application.

The packaged applications do not require a separate Python installation or additional dependency setup.

> **macOS note:** The application can be launched directly from the extracted folder; moving it to the **Applications** folder is optional.  
> On first launch, macOS may block the application because it is not notarized by Apple. If this occurs, open **System Settings → Privacy & Security**, then select **Open Anyway** to authorize the application.

## Basic Usage

1. Select an experimental MS/MS peak-list file.
2. Preview the input data and confirm the `m/z` and intensity columns.
3. Adjust ppm, noise, S/N, and diagnostic-ion thresholds if necessary.
4. Run the analysis.
5. Inspect ranked candidates, diagnostic ions, and experimental/predicted spectra.
6. Export the results to an Excel workbook.

## Output

Analysis results can be exported as an `.xlsx` workbook containing:

* `Parameters` — analysis and scoring parameters
* `Candidate_Ranking` — ranked structural candidates and HDCS results
* `Diagnostic_Ions` — theoretical and experimental diagnostic-ion matches

The graphical interface also provides an interactive butterfly plot comparing experimental and predicted EAciD-MS/MS spectra.

## Notes

This software is intended for **research use** in TG-FAHFA structural annotation.

The current fragmentation rules, candidate space, precursor library, and scoring parameters were developed for the associated EAciD-MS/MS workflow. Default parameters may require optimization for different instruments or acquisition conditions.

Candidate ranking should be interpreted together with the underlying diagnostic-ion evidence. NNLS contributions are supporting evidence for potential mixtures and should not be directly interpreted as quantitative molar fractions.

## Citation

If you use this software in academic research, please cite the associated publication describing the TG-FAHFA EAciD-MS/MS and HDCS workflow.

> Citation information will be added after publication.

## License

This project is licensed under the **BSD 3-Clause License**.

Copyright © 2026 **WTUFengGroup**.

Redistribution and use in source and binary forms, with or without modification, are permitted subject to the conditions of the BSD 3-Clause License.

See the [`LICENSE`](LICENSE) file for the complete license text.

## Contact

For questions, bug reports, or feature requests, please use the GitHub **Issues** page.
