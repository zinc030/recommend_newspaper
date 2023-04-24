#오아연 2016106125

import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import gensim
import nltk.data
import nltk
import newspaper
from newspaper import Config
import tqdm
from tqdm import tqdm


def collect_news():
    driver_path = "C:/Users/zinc/chromedriver_win32/chromedriver.exe"
    driver = selenium.webdriver.Chrome(driver_path)

    driver.get('https://news.google.com/home?hl=en-US&gl=US&ceid=US:en')

    wait = WebDriverWait(driver, 10)
    # 특정 a태그가 나올 때까지 기다림
    #wait.until(EC.visibility_of_element_located((By.XPATH, '//a[text()="Top stories"]')))

    #top stories 링크로 클릭해서 들어간 뒤, 특정 h2태그가 나올따까지 기다림
    driver.find_element(By.CLASS_NAME, "aqvwYd").click()
    wait.until(EC.visibility_of_element_located((By.XPATH, '//h2[text()="Headlines"]')))

    article_nodes = driver.find_elements(By.TAG_NAME, "article")
    df = pd.DataFrame(columns=['title', 'source', 'link'])

    for article in tqdm(article_nodes):
        title = article.find_element(By.XPATH, ".//h4").text
        link = article.find_element(By.XPATH, ".//div/a").get_attribute('href')
        source = article.find_element(By.XPATH, ".//div/span").text

        # 제목, 출처, 링크 수집해서 pandas의 dataframe에 저장
        df = pd.concat([df, pd.DataFrame({"title": title, "source": source, "link": link}, index=[0])],
                       ignore_index=True)

    driver.quit()
    return df


def fill_news_contents(df):
    # newspaper 링크 주소를 가져와서 다운하고 parse하면, 각 해당 링크의 newspaper의 타이틀과 dataframe에 수집할 수 있음.
    config0 = Config()
    config0.request_timeout = 10 # 패킷이 목적지에 도착하지 않았을 경우? 너무 오래 기다리는 것을 방지하기 위해 10초로 설정

    df = df.assign(text=None)

    for index, row in tqdm(df.iterrows()):
        try:
            news = newspaper.Article(row['link'], language='en', config=config0) # request_timeout
            news.download()
            news.parse()

            title = row['title'].strip() if row['title'].strip() else news.title
            contents = news.text.strip() if news.text.strip() else title  #원래 else news.title이였는데 else title로 바꿈
            df.at[index, 'title'] = title
            df.at[index, 'text'] = contents
        except Exception as e:
            df.at[index, 'text'] = df.at[index, 'title']
            print(e)
            pass

    # 강제구독창 또는 동영상 웹사이트 때문에 제목/내용이 없는 row 삭제
    # for index, row in df.iterrows():
    #     if not row['title']:
    #         df.drop(index, inplace=True)
    # df.reset_index(drop=True, inplace=True) # df의 index가 다시 0부터 시작

    return df


def train_news(df):
    # 뉴스 데이터로 모델 학습해서 모델 반환
    nltk.download("punkt")
    nltk.download('stopwords')

    stop_words = nltk.corpus.stopwords.words('english') # 'and', 'or', 'with' 등 자주 사용되지만 중요하지 않은 단어들 (근데 제거 안 하는게 더 좋을 수도 있을까?)

    preprocessed_articles = []

    for article in df["text"].tolist():
        words = [
            word.lower()
            for sentence in nltk.sent_tokenize(article)
            for word in nltk.word_tokenize(sentence)
            if word.lower() not in stop_words
        ]
        tokenized_articles = words
        preprocessed_articles.append(tokenized_articles) # preprocessed_articles list에는 뉴스마다 리스트가 있고 그 리스트에는 tokenize된 단어들이 들어감

    # 불필요한 (특수)문자 정리! 정리가 필요할까? 근데 왜인지 한번 돌려도 정리가 안된 element가 많음.
    exclude_chars = ['.', ',', '-', '"', '\'', '!', '~', '‘', ':', ';', '—', '&', '|', '?', '#', '[', ']', '(', ')', '$']
    for arti in preprocessed_articles:
        for words in arti:
            if any(char in words for char in exclude_chars):
                arti.remove(words)
    # print(preprocessed_articles)

    model = gensim.models.Word2Vec(preprocessed_articles, vector_size=100, window=5, min_count=5, workers=4)  # window=예측하는 단어까지의 최대 거리, min_count=빈도수 5보다 낮으면 단어 무시
    return model, preprocessed_articles


def choose_article(df, model, preprocessed_articles, tag):
    # 학습된 모델을 가지고 유저의 관심 키워드 와 가장 비슷한 뉴스를 비교를 통해 찾아 주는 함수. Parameter 는 dataframe, 학습된 word2vec 모델, preprocessed articles, 그리고 태그
    if tag not in model.wv.index_to_key: #similarity 돌리기 전에 tag가 훈련된 모델의 단어장에 존재하지 않으면 문장 출력하면서 종료
        print(f"there is no articles about {tag}")
        return

    interest_tag_vector = model.wv[tag] #모델 단어장과 일치하는 tag 단어의 관련 수치 100개 있는 vector list
    # print(interest_tag_vector) #test

    article_vectors = []
    for article in preprocessed_articles:
        article_vector = sum([model.wv[word] for word in article if word in model.wv.index_to_key])
        article_vectors.append(article_vector)
    # print(article_vectors) #test

    similarities = {}
    for i, article_vector in enumerate(article_vectors):
        similarity = model.wv.cosine_similarities(article_vector, [interest_tag_vector])[0]
        similarities[i] = similarity
    # print(similarities) #test

    sorted_articles = sorted(similarities, key=lambda x: similarities[x], reverse=True)
    article_title = df.at[sorted_articles[0], "title"]
    article_link = df.at[sorted_articles[0], "link"]
    print(f"The most similar article for '{tag}' is '{article_title}' at {article_link}")

# print(collect_news().to_string()) #test
df = fill_news_contents(collect_news())
# print(df.to_string()) #test
model, preprocessed_articles = train_news(df)

while True:
    try:
        tag = input("Enter a tag (to quit, enter -1): ") #종료하고 싶으면 -1, 안그럼 계속 기사 추천 받을 수 있음
        if tag == "-1":
            break
        else:
            tag = tag.lower() #같은 단어여도 대문자가 들어있으면 감지를 못 함
            choose_article(df, model, preprocessed_articles, tag)
    except:
        print("There was an error. Please enter another word") #오류나면 출력


