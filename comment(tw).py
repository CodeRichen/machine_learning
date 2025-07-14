import json
import re
import os
from collections import Counter
import jieba
from snownlp import SnowNLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from googletrans import Translator
import matplotlib.pyplot as plt

translator = Translator()

def translate_to_chinese(text):
    try:
        result = translator.translate(text, dest='zh-tw')
        return result.text
    except Exception as e:
        print(f"翻譯失敗: {e}")
        return text

def preprocess_text(text):
        # 保留 中文、英文、數字、日文平假名、片假名、部分日文符號
    return re.sub(r'[^\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff\u3000-\u303fA-Za-z0-9]', '', text)

# 停用詞清單
stop_words = [
    '無法', '什麼', '一首', '可以', '這是', '首歌', '仍然', '但是', '聲音',
    '電子', '一起', '如此', '最好', '喜歡', '評論', '很棒', '知道', '歌曲', '認為', '上面'
]


# Tokenizer：只保留長度 >1 且不在停用詞中的詞
def custom_tokenizer(text):
    tokens = jieba.lcut(text)
    return [token for token in tokens if len(token) > 1 and token not in stop_words]

# === 1. 下載留言 ===
video_url = input("請輸入 YouTube 影片網址：")
output_file = './output/comment.json'
os.system(f'youtube-comment-downloader --url {video_url} --output {output_file}')

# === 2. 讀取並去除重複留言（只保留第一筆）===
# === 2. 讀取並根據作者與文字內容進行過濾（結合作者與內容判斷）===
comment_by_author = {}  # author -> 最短的留言資料
seen_texts = set()
final_comments = []

with open(output_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        author = data.get('author', '')
        text = data['text'].strip()

        # 已出現過的留言內容（無論誰說的）就跳過
        if text in seen_texts:
            continue
        seen_texts.add(text)

        # 如果是這個作者第一次出現，直接加
        if author not in comment_by_author:
            comment_by_author[author] = data
        else:
            # 若目前這句更短，則更新作者的留言
            prev_text = comment_by_author[author]['text'].strip()
            if len(text) < len(prev_text):
                comment_by_author[author] = data
            # 若長度相同就不更新（保留第一筆）

# 最後整理符合條件的留言
final_comments = list(comment_by_author.values())

# 寫回檔案
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in final_comments:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

print(f"去重後留言數（根據作者與文字）：{len(final_comments)}")


# === 3. 翻譯 + 情緒分析（僅第6~8、49~51）===
comments = []
translated_comments = []
enriched_comments = []

# 翻譯限制條件
should_translate_all = len(final_comments) <= 500

# 排序找目標 index
sorted_indices = sorted(range(len(final_comments)), key=lambda i: len(final_comments[i]['text']), reverse=True)
top_indices = sorted_indices[5:8]  # 第6~8則

length_sorted = sorted(final_comments, key=lambda d: len(d['text']))
mid_start = max(len(length_sorted) // 2 - 1, 0)
mid_indices = [i for i, d in enumerate(final_comments) if d['text'] == length_sorted[mid_start]['text'] or
               d['text'] == length_sorted[mid_start + 1]['text'] or
               d['text'] == length_sorted[min(mid_start + 2, len(length_sorted)-1)]['text']]

target_indices = set(top_indices + mid_indices)

for i, data in enumerate(final_comments):
    text = data['text']
    # 若總數小於等於 500 則全部翻譯；否則僅翻譯目標句（若有非中文）
    should_translate = should_translate_all or (i in target_indices and re.search(r'[^\u4e00-\u9fa5]', text))
    translated = translate_to_chinese(text) if should_translate else text

    s = SnowNLP(translated)
    sentiment_score = round(s.sentiments, 3)

    enriched_comments.append({
        'original_text': text,
        'translated_text': translated,
        'sentiment_score': sentiment_score
    })

    comments.append(text)
    translated_comments.append(translated)


# === 4. 儲存 enriched 結果 ===
with open('./output/comments_translated_sentiments.json', 'w', encoding='utf-8') as f:
    json.dump(enriched_comments, f, ensure_ascii=False, indent=2)

# === 5. 詞頻分析 ===
word_counter = Counter()
for comment in comments:
    words = jieba.lcut(preprocess_text(comment))
    filtered = [w for w in words if len(w) > 1 and w not in stop_words]
    word_counter.update(filtered)

print("\n最常見詞前20：")
for word, count in word_counter.most_common(20):
    print(f"{word}: {count}")

# === 6. 情緒分析輸出 ===
print("\n情緒分析 - 長度第6~8留言：")
for idx in top_indices:
    c = enriched_comments[idx]
    print(f"留言：{c['original_text']}\n翻譯：{c['translated_text']}\n情緒分數：{c['sentiment_score']}\n")

print("\n情緒分析 - 長度中間第49~51留言：")
for idx in mid_indices[:3]:
    c = enriched_comments[idx]
    print(f"留言：{c['original_text']}\n翻譯：{c['translated_text']}\n情緒分數：{c['sentiment_score']}\n")

# === 7. 主題模型（LDA）===
processed_comments = [preprocess_text(c) for c in comments]
vectorizer = CountVectorizer(
    max_df=0.9,
    min_df=5,
    tokenizer=custom_tokenizer,
    stop_words=None
)
X = vectorizer.fit_transform(processed_comments)
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

print("\n主題模型 LDA：")
for idx, topic in enumerate(lda.components_):
    top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
    print(f"主題 {idx + 1}: {top_words}")

# === 8. 詞雲 ===
all_comments_text = ' '.join(translated_comments)
tokens = [w for w in jieba.lcut(preprocess_text(all_comments_text)) if len(w) > 1 and w not in stop_words]
filtered_text = ' '.join(tokens)

wordcloud = WordCloud(
    font_path='msjh.ttc',
    width=1200,
    height=800,
    background_color='white',
    max_font_size=120,
    min_font_size=10,
    colormap='viridis'
).generate(filtered_text)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Original Comments')
plt.show()
