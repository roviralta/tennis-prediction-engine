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

## ATP archive and daily feed

The working source is the local Jeff Sackmann ATP mirror under
`data/external/tennis-sackmann-archive/atp`. The last verified report showed
match dates through 2026-05-25 and ranking dates through 2026-06-08. Its historical
player IDs and dated rankings are the migration baseline.
[Public mirror](https://github.com/Aneeshers/tennis-sackmann-archive)

TennisMyLife is the planned daily match source. Its website documents a
file-list API at `https://stats.tennismylife.org/api/data-files` and CSV downloads
for personal, educational, and non-commercial use. Use ATP singles results
and verify match status, surface, player IDs, score, and available statistics.
TennisMyLife uses different player identifiers, so source IDs need an explicit
mapping. Verify a separately dated ranking feed and do not infer ranking
publication dates from the time a CSV was downloaded.
[TennisMyLife database and terms](https://stats.tennismylife.org/tennis-match-database)
