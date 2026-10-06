from tennis_prediction.ingestion.atp.loader import ATPDataLoader


loader = ATPDataLoader(
    "data/external/tennis-sackmann-archive/atp"
)

matches = loader.load_matches(2025, 2025)

print(matches.head())
print()
print(f"Rows: {matches.height}")
print(f"Columns: {matches.width}")
print()
print(matches.columns)
