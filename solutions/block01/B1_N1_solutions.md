# Questions

## Section 1.1: Exploring image level differences

- Do the overall image intensities appear similar?

    Between the raw images, the intensities appear to be very different, with different ROIs displaying different contrasts as well. In the images registered to standard space, the intensities and contrasts are slightly improved both visually and empirically, but there is still a noticable difference.

- Are some images noisier than others?

    While we have not specifically analysed the SNR and CNR here, there are distinct visual differences between the images. In real studies, where the protocols, scanners and individuals may be different, this difference would be more severe

- Does tissue contrast differ between scans?

    Small differences in tissue contrast are often visible, particularly at the boundaries between grey matter, white matter, and cerebrospinal fluid. These contrast differences may influence how easily tissues can be distinguished by both human observers and automated analysis tools. This is an espeically important concept to be aware of for analysis of Imaging Derived Phenotypes (IDPs) describing brain structure.

- Are anatomical boundaries equally clear?

    In this example, most anatomical boundaries are fairly clear between the two images, however the relative size, magnitude of folding on the cortex and other structural features do show distinct differences

- Could these differences influence automated measurements such as hippocampal volume or cortical thickness?

    Yes. Many neuroimaging analysis pipelines rely on image intensity and tissue contrast to identify anatomical structures. Scanner-related differences can therefore affect segmentation accuracy and lead to variability in derived measurements such as regional brain volumes, cortical thickness, or tissue properties.

## Section 1.2: Basic image level statistics

- What can the intensity histograms tell us and is there anyway to make them more informative?

    They not only tell us the range of values, but also the spread of values within an image. Substantially different histograms would mean substantially different contrasts. The current plot is of the whole brain including outside of the skull, removing the middle peak may make it more informative. This plot can also be used on an ROI by ROI basis to tell you how the intensity changes across the brain.

- What can the total number of non-zero voxels tell us?

    The number of non-zero voxels can tell us two things mainly; The relative size of the brain compared to total data size, the amount of residual noise outside of the brain. Some scanners and image processing pipelines will use masks to set the data outside of the brain to zero and others may not, meaning we cannot always trust this result.

## Section 1.3: ROI/IDP differences due to scanner effects:

- Looking at the data and the relative size of the bars, make some notes in things you notice

    The main thing to notice is that the order of scanners isn't concistent between different measures, showing that while there is a clear difference, this difference isn't always the same in magnitude nor direction across the brain.

