from pathlib import Path
import yfinance as yf

project_root = Path(__file__).resolve().parent.parent.parent
raw_dir = project_root / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

df = yf.download(
    "SPY",
    period="60d",
    interval="2m",
    auto_adjust=False,
    progress=False,
    prepost=False
)

df = df.reset_index()
df.to_csv(raw_dir / "spy_2m.csv", index=False)

print(df.head())
print(df.shape)