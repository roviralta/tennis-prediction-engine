## Data Quality: Playing Surface

The ATP historical dataset contains four known surface categories:

- `Hard`
- `Clay`
- `Grass`
- `Carpet`

Some matches have a missing `surface` value. An initial inspection of the full dataset found 2,990 matches with an unknown surface out of 199,389 matches.

Missing values are concentrated mainly in the earliest historical records, although some recent matches also have missing surface information.

### Data handling policy

- Preserve matches with missing surface information.
- Do not infer or fabricate missing surface values.
- Keep `Carpet` as a distinct surface category.
- Handle missing values during model-specific preprocessing.
- Evaluate model performance across different historical periods and surface categories.

Missing surface information is a data-quality limitation, not an automatic reason to discard a match.