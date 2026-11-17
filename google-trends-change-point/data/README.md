# Input data

`multiTimeline.csv` is an unchanged copy of the supplied Google Trends export
`multiTimeline (1).csv`. Only the filename has been simplified.

| Property | Value |
| --- | --- |
| Series | `mask: (United States)` |
| Category, as recorded in the file | All categories |
| Frequency | Monthly |
| Coverage | January 2004–February 2025, inclusive |
| Observations | 254 |
| Observed values | Integers from 4 to 100 |
| Missing entries | None |

The first line is `Category: All categories`, the second line is blank, and
the third line contains the column names `Month,mask: (United States)`.
The script therefore reads the file with `skiprows=2`.

The inference uses the search-interest column in its original order. The plot
uses the `Month` column, formatted as `YYYY-MM`, for calendar labels. The
month sequence is complete and chronological. No values are filtered,
imputed, rescaled, or otherwise transformed.

The export does not record its download timestamp, an export URL, or every
Google Trends setting. Those details have not been inferred. Reuse this
included file to reproduce the displayed results.
