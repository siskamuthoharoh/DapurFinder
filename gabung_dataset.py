
import pandas as pd
from pathlib import Path

# Lokasi folder dataset
folder = Path(r"D:\STKI-Recipe-Finder\dataset")

# Nama delapan file CSV
daftar_file = [
    "dataset-ayam.csv",
    "dataset-ikan.csv",
    "dataset-kambing.csv",
    "dataset-sapi.csv",
    "dataset-tahu.csv",
    "dataset-telur.csv",
    "dataset-tempe.csv",
    "dataset-udang.csv",
]

semua_data = []

# Membaca setiap file CSV
for nama_file in daftar_file:
    lokasi_file = folder / nama_file

    df = pd.read_csv(lokasi_file)

    # Menambahkan kategori berdasarkan nama file
    kategori = nama_file.replace("dataset-", "").replace(".csv", "")
    df["Category"] = kategori

    semua_data.append(df)
    print(f"Berhasil membaca: {nama_file} ({len(df)} resep)")

# Menggabungkan semua data
recipes = pd.concat(semua_data, ignore_index=True)

# Membersihkan teks dan menghapus judul kosong
for kolom in ["Title", "Ingredients", "Steps"]:
    recipes[kolom] = recipes[kolom].fillna("").astype(str).str.strip()

recipes = recipes[recipes["Title"] != ""]
recipes = recipes.drop_duplicates().reset_index(drop=True)

# Menyimpan hasil di folder dataset
output = folder / "recipes.csv"
recipes.to_csv(output, index=False, encoding="utf-8-sig")

print("\n=== HASIL PENGGABUNGAN ===")
print("Jumlah resep:", len(recipes))
print("Jumlah kolom:", len(recipes.columns))
print("Nama kolom:", recipes.columns.tolist())
print("Jumlah resep per kategori:")
print(recipes["Category"].value_counts())
print("\nFile berhasil disimpan di:", output)
