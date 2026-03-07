import pandas as pd

path = "archive/detect_cuneiform_transliteration/page.csv"
df = pd.read_csv(path)
MARKERS = {"DUMU", "LUGAL", "KÙ.BABBAR", "KUBABBAR", "URU", "É", "KUR", "KIIB", "DAM.QAR"}
DIACRITICS = set('šṣṭḫāēīūáíúà')

print(df.columns)
countList = []
propList = []
df["text"] = df["text"].astype(str)
for idx, row in df.iterrows():
    txtList = row["text"].split()
    counter = 0
    for i in range(len(txtList)):
        for ch in txtList[i]:
            if (0x12000 <= ord(ch) <= 0x123FF or 0x12400 <= ord(ch) <= 0x1247F or 0x12480 <= ord(ch) <= 0x1254F):
                counter+=1
                break
    countList.append(counter)
    propList.append(counter/len(txtList))

df["words_in_cuneiform"] = countList
df["proportion_cuneiform"] = propList
df.to_csv(path.split(".")[0] + "_with_counts_and_proportions.csv")