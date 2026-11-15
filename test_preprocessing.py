
import pandas as pd

from modules.preprocessing import (
    run_preprocessing
)


# Dataset pengujian sederhana
data = {

    "id_str": [
        "100001",
        "100002",
        "100003",
        "100003"
    ],

    "user_id_str": [
        "akun_a",
        "akun_b",
        "akun_c",
        "akun_c"
    ],

    "full_text": [
        "MBG SANGAT BAGUUUUS!!! #MBGIndonesia 👍",
        "@akun123 Program MBG sangat bagus https://contoh.com",
        "Program MBG ini kurang baguuuuus!!!",
        "Program MBG ini kurang baguuuuus!!!"
    ]
}

df_test = pd.DataFrame(data)


# Jalankan preprocessing
df_hasil, summary = run_preprocessing(
    df_test
)


# Tampilkan hasil
print("=" * 60)
print("HASIL TEST PREPROCESSING")
print("=" * 60)

print("\nRingkasan:")
print(summary)

print("\nHasil teks:")

print(
    df_hasil[
        [
            "full_text",
            "clean_text"
        ]
    ].to_string(index=False)
)
