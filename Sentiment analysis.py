"""
This program is wrote to be runed in Python Interactive Window. 
So each couple of lines can be runed together without the needing to run all the script.

"""
from sqlalchemy import create_engine
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk import WordNetLemmatizer
from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import re
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from wordcloud import WordCloud



def label_kaggle_reviews(): 
    df_kaggle= pd.read_csv('Hotel_Reviews.csv')
    reviews_dic = {"reviews":[], "label":[]}
    for negative, positive in zip(df_kaggle["Negative_Review"], df_kaggle["Positive_Review"]):
        if positive == 'No Positive' or negative == "No Negative" or len(negative.split())<=1 or len(positive.split())<=1:
            pass
        #Remove N a words, and remove the review if the review consist of two words en each word is one letter
        elif((len(negative.split())<=2 and (len(negative.split()[0])<=1 and len(negative.split()[1])<=1)) or (len(positive.split())<=2 and (len(positive.split()[0])<=1 and len(positive.split()[1])<=1))):
            pass
        else:
            [reviews_dic["reviews"].append(negative), reviews_dic["label"].append(0)]
            [reviews_dic["reviews"].append(positive), reviews_dic["label"].append(1)]

    df_kaggle_reviews = pd.DataFrame(reviews_dic)
    return df_kaggle_reviews


def load_slice_dataframes():
    engine = create_engine('mysql+mysqlconnector://root:*****@127.0.0.1:3306/test').connect() 
    df_kaggle_reviews = label_kaggle_reviews()
    gescrapt_reviews = pd.read_sql_table('reviews_scraped', engine)
    self_wrote_reviews= {"reviews":["very bad hotel","very good hotel", "nice hotel and the view of the hotel is also great"],
                        "label":[0,1,1]}
    df_self_wrote = pd.DataFrame.from_dict(self_wrote_reviews) 
    
    df = pd.concat([df_kaggle_reviews, gescrapt_reviews,df_self_wrote],axis=0)

    dfFilter_pos = df.loc[df['label']== 1][:25000]
    dfFilter_neg = df.loc[df['label']== 0][:25000] 

    df_reviews = pd.concat([dfFilter_pos, dfFilter_neg],axis=0)
    df_reviews = df_kaggle_reviews.sample(frac=1).reset_index(drop=True)

    df_reviews.to_csv (r'.\labeled_data.csv', index = False, header=True)
    df_reviews.to_sql(name='all reviews', con=engine, if_exists='replace', index=False, method='multi')
    # return df_kaggle_reviews


def preprocess_reviews(reviews):
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
    # REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    NO_SPACE = ""
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    return [' '.join([word for word in review.split() if word.isalpha()]) for review in reviews]


def get_stemmed_text(corpus):
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]


def get_lemmatized_text(corpus): 
    import nltk
    nltk.download('wordnet') 
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]


def word_cloud_():
    stop_words = ENGLISH_STOP_WORDS.union(("didn",'thing','really','door','time','just','day','and','or','if',"did",'hotel',"No Positive","No Negative"))
    strin = ''
    for revi in df_kaggle_reviews.reviews:
        strin = strin + ', '+revi

    my_cloud_2 = WordCloud(background_color='white',stopwords=stop_words).generate(strin)
    plt.imshow(my_cloud_2, interpolation='bilinear')
    plt.axis("off")



def make_plot(columnsName, xlabel, ylabel,title):
    df_kaggle= pd.read_csv('Hotel_Reviews.csv')
    df_kaggle = df_kaggle[columnsName].value_counts()[:10].sort_values().tail(15)

    ax = df_kaggle.plot(kind='barh', figsize=(4, 6), color='#86bf91', zorder=2, width=0.85)

    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Switch off ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw vertical axis lines
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel(xlabel, labelpad=20, weight='bold', size=12)
    ax.set_ylabel(ylabel, labelpad=20, weight='bold', size=12)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

#Run wen you want to shuffle and remake a CSV file.
# load_slice_dataframes()

df_kaggle_reviews = pd.read_csv('.\labeled_data.csv')
df_kaggle_reviews['reviews_stem'] = get_stemmed_text(preprocess_reviews(df_kaggle_reviews.reviews))
my_stop_words = get_stemmed_text(ENGLISH_STOP_WORDS.union(('and','or','if')))
# my_pattern = r'\b[^\d\W][^\d\W]+\b' #token_pattern=


# # TFIDF frequnsie of the words so how many times each word is apper in the string
# vect_Tfid = TfidfVectorizer(ngram_range=(1, 2), max_features=100,
#                         stop_words=my_stop_words).fit(df_kaggle_reviews['reviews_stem'])
# X_txt_Tfid = vect_Tfid.transform(df_kaggle_reviews['reviews_stem'])
# df_Tfid = pd.DataFrame(X_txt_Tfid.toarray(), columns=vect_Tfid.get_feature_names())


# min_df ( = 5): defines the minimum frequency of a word for it to be counted as a feature

vect_BOW = CountVectorizer(ngram_range=(1,2), stop_words=my_stop_words, max_features=2000, binary=True).fit(df_kaggle_reviews.reviews_stem)
X_txt_BOW = vect_BOW.transform(df_kaggle_reviews.reviews_stem)
df_BOW = pd.DataFrame(X_txt_BOW.toarray(), columns=vect_BOW.get_feature_names())


df_BOW['label'] = df_kaggle_reviews.label
y = df_BOW.label
X = df_BOW.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

models = {"BernoulliNB": BernoulliNB(),
 "LogisticRegression": LogisticRegression(),
  "LinearSVC":  LinearSVC(),
   "RandomForestClassifier": RandomForestClassifier() 
}
for model_name in models:
    models[model_name].fit(X_train, y_train)
    # models[model_name].predict(X_test)





reviews = ['The breakfast was so nothing. completely disappointing.',
     "Everything was ok for me . The staff were lovely.",
     "The Hotel was clean and the bed comfortable. The staff were very friendly."]

for model_name in models:
    print("{} resultaat: {}".format(model_name, models[model_name].predict(vect_BOW.transform(get_stemmed_text(preprocess_reviews(reviews))))))
    
    print('Accuracy of model {}: \n {}'.format(model_name,models[model_name].score(X_test, y_test)))
    print('Accuracy of model {}: \n {}'.format(model_name, accuracy_score(y_test, models[model_name].predict(X_test))))

    # Print performance metrics for each model
    print('Confusion matrix of model {}: \n {}'.format(model_name, confusion_matrix(y_test, models[model_name].predict(X_test))/len(y_test)))

    plot_confusion_matrix(models[model_name], X_test, y_test) 
    plt.show()  




param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)


make_plot(columnsName="Hotel_Name",xlabel="Aantal reviews", ylabel="Aantal reviews voor elke hotel ", title='Aantal reviews voor elke Hotel')
make_plot(columnsName="Reviewer_Nationality",xlabel="Aantal reviews", ylabel="Aantal reviews voor elke Nationality", title='Aantal reviews voor elke Nationality')
word_cloud_()
