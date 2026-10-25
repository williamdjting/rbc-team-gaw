# Preprocessing results

The two archives were processed with the same filters. The complete output contains **884,963 May–September BC observations from 2018–2024**. Splitting into development and holdout periods is intentionally postponed.

The new archive `DL_FIRE_SV-C2_830406.zip` contains 260,798 observations from 2018–2020. The existing archive contributes the 2021–2024 records. Both use the same 15-column FIRMS schema.

Applying May–September and BC point-centre filters to both archives produced **884,963 observations**:

| Year | BC summer observations |
| ---: | ---: |
| 2018 | 221,629 |
| 2019 | 4,629 |
| 2020 | 3,482 |
| 2021 | 128,440 |
| 2022 | 17,077 |
| 2023 | 376,650 |
| 2024 | 133,056 |

31,386 observations were outside the BC boundary after the summer and year filters. The output has zero null cells, zero exact duplicate rows within processed chunks, and zero invalid coordinate rows.

The year imbalance remains substantial: 2023 contributes about 42.6% of the merged rows, while 2019 and 2020 are sparse. More years improve seasonal coverage, but source coverage and satellite observation differences still need review before treating every year as equally comparable. Splits should be chosen after this merged-data review.

