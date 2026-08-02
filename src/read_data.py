import pandas as pd

cricket = pd.read_excel("data/World_Cricketers.xlsx")
olympics = pd.read_excel("data/Indian_Olympic_Players.xlsx")

print("\n===== CRICKET DATA =====")
print(cricket.head())
print("\nColumns:")
print(cricket.columns.tolist())

print("\n==============================")

print("\n===== OLYMPICS DATA =====")
print(olympics.head())
print("\nColumns:")
print(olympics.columns.tolist())