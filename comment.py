# -*- coding: utf-8 -*-
import os
import re
import json
import regex
import jieba
import matplotlib.pyplot as plt
from snownlp import SnowNLP
from collections import Counter
from googletrans import Translator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from fugashi import Tagger

# === 初始化 ===
translator = Translator()
tagger = Tagger()

# 停用詞
stop_words = ['無法', '什麼', '一首', '可以', '這是', '首歌', '仍然', '但是', '聲音',
              '電子', '一起', '如此', '最好', '喜歡', '評論', '很棒', '知道', '歌曲', '認為', '上面']
english_stopwords = {
    'this', 'is', 'the', 'a', 'an', 'it', 'that', 'of', 'to', 'and', 'for', 'in', 'on', 'with',
    'as', 'i', 'you', 'he', 'she', 'they', 'we', 'me', 'my', 'your', 'his', 'her', 'its',
    'our', 'their', 'be', 'was', 'were', 'are', 'am', 'have', 'has', 'do', 'did', 'at',
    'by', 'from', 'but', 'not', 'or', 'so', 'what', 'can', 'just', 'will', 'about'
}
japanese_stopwords = [
    'これ', 'それ', 'あれ', 'どれ', 'ここ', 'そこ', 'あそこ', 'こちら', 'どこ',
    'わたし', 'あなた', 'かれ', 'かのじょ', 'わたしたち', 'みんな',
    'は', 'が', 'を', 'に', 'で', 'と', 'も', 'の', 'よ', 'ね', 'から', 'まで', 'へ', 'など', 'や', 'より', 'って',
    'です', 'ます', 'でした', 'ません', 'でしょう', 'だろう', 'たい', 'ない', 'たく', 'たち',
    'そして', 'それから', 'しかし', 'だから', 'でも', 'つまり', 'ところで', 'ただし', 'また', 'けど', 'ちなみに', 'さて', 'まあ', 'なんか',
    'ああ', 'うん', 'ええ', 'えっ', 'うわー', 'はい', 'ほんとに', 'すごい', 'やばい', 'まじ', 'なんと', 'どうも'
]
all_stopwords = set(stop_words + list(japanese_stopwords) + list(english_stopwords))

# === 工具函數 ===
def translate_to_chinese(text):
    try:
        return translator.translate(text, dest='zh-tw').text
    except Exception as e:
        print(f"翻譯失敗: {e}")
        return text

def preprocess_text(text):
    return regex.sub(r'[^\p{Han}\p{Hiragana}\p{Katakana}a-zA-Z0-9\s]', '', text)

def extract_chinese_japanese_words(text):
    chinese_words = jieba.lcut(text)
    japanese_words = [w.surface for w in tagger(text)]
    combined = chinese_words + japanese_words
    return [w for w in combined if len(w.strip()) > 1 and w.lower() not in all_stopwords]

# === 1. 下載留言 ===
video_url = input("請輸入 YouTube 影片網址：")
output_file = './output/comment.json'
os.makedirs('./output', exist_ok=True)
os.system(f'youtube-comment-downloader --url {video_url} --output {output_file}')

# === 2. 過濾重複留言 ===
comment_by_author = {}
seen_texts = set()
with open(output_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        author = data.get('author', '')
        text = data['text'].strip()
        if text in seen_texts:
            continue
        seen_texts.add(text)
        if author not in comment_by_author or len(text) < len(comment_by_author[author]['text']):
            comment_by_author[author] = data
final_comments = list(comment_by_author.values())
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in final_comments:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')
print(f"✅ 去重後留言數：{len(final_comments)}")

# === 3. 情緒分析 ===
comments = []
translated_comments = []
top_sentiment_results = []
mid_sentiment_results = []

sorted_indices = sorted(range(len(final_comments)), key=lambda i: len(final_comments[i]['text']), reverse=True)
top_indices = sorted_indices[5:8]
length_sorted = sorted(final_comments, key=lambda d: len(d['text']))
mid_start = max(len(length_sorted) // 2 - 1, 0)
mid_texts = {length_sorted[mid_start]['text'], length_sorted[mid_start + 1]['text'], length_sorted[min(mid_start + 2, len(length_sorted)-1)]['text']}
mid_indices = [i for i, d in enumerate(final_comments) if d['text'] in mid_texts]
target_indices = set(top_indices + mid_indices)

for i, data in enumerate(final_comments):
    text = data['text']
    comments.append(text)
    if i in target_indices:
        translated = translate_to_chinese(text) 
        sentiment_score = round(SnowNLP(translated).sentiments, 3)
        result = {'original_text': text, 'translated_text': translated, 'sentiment_score': sentiment_score}
        (top_sentiment_results if i in top_indices else mid_sentiment_results).append(result)
    translated_comments.append(text)

print("\n情緒分析 - 長度第6~8名留言：")
for c in top_sentiment_results:
    print(f"留言：{c['original_text']}\n翻譯：{c['translated_text']}\n情緒分數：{c['sentiment_score']}\n")

print("\n情緒分析 - 中間留言：")
for c in mid_sentiment_results:
    print(f"留言：{c['original_text']}\n翻譯：{c['translated_text']}\n情緒分數：{c['sentiment_score']}\n")

# === 4. 詞頻分析（中日混合） ===
word_counter = Counter()
for comment in comments:
    cleaned = preprocess_text(comment)
    tokens = extract_chinese_japanese_words(cleaned)
    word_counter.update(tokens)

print("\n最常見詞前20:")
for word, count in word_counter.most_common(20):
    print(f"{word}: {count}")

# === 5. LDA 主題模型 ===
processed_comments = [' '.join(extract_chinese_japanese_words(preprocess_text(c))) for c in comments]
try:
    vectorizer = CountVectorizer(max_df=0.9, min_df=5)
    X = vectorizer.fit_transform(processed_comments)
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)

    print("\n主題模型 LDA：")
    for idx, topic in enumerate(lda.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        print(f"主題 {idx + 1}: {top_words}")

except ValueError as e:
    print(f" LDA 執行失敗：{e}")
    print("嘗試將 min_df 降為 2 再試一次...")
    vectorizer = CountVectorizer(max_df=0.9, min_df=2)
    X = vectorizer.fit_transform(processed_comments)
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)

    print("\n主題模型 LDA（降 min_df=2）：")
    for idx, topic in enumerate(lda.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        print(f"主題 {idx + 1}: {top_words}")


# === 6. 詞雲 ===
all_text = ' '.join(translated_comments)
tokens = extract_chinese_japanese_words(preprocess_text(all_text))
filtered_text = ' '.join(tokens)

# 字型偵測（自動載入第一個存在的）
font_candidates = [
    "./ipaexm.ttf",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/msmincho.ttc",
    "C:/Windows/Fonts/NotoSansCJKjp-Regular.otf"
]
font_path = next((f for f in font_candidates if os.path.exists(f)), None)
if not font_path:
    raise FileNotFoundError("error")

wordcloud = WordCloud(
    font_path=font_path,
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
plt.title('Word Cloud of Original Comments (with Japanese)')
plt.show()
