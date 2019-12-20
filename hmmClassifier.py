from seqlearn import hmm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from bs4 import BeautifulSoup
import re
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

train_path = "TrainingSentence1000.tsv"
test_path = "TestSentence.tsv"


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    try:
        string = re.sub(r'\\', "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        return string.strip().lower()
    except:
        print(string)


vectorizer = TfidfVectorizer(min_df=5,
                             max_df=0.8,
                             sublinear_tf=True,
                             use_idf=True)

data_train = pd.read_csv(train_path, sep='\t', encoding='ISO-8859-1')
train_vectors = vectorizer.fit_transform(data_train.review.values.astype('U'))

model = hmm.MultinomialHMM()

lengthOfModelfit = [1 for i in range(len(data_train.review)-10)]
# lengthOfModelfit[0] = 2
model = model.fit(train_vectors, data_train.sentiment, lengthOfModelfit)


data_test = pd.read_csv(test_path, sep='\t', encoding='ISO-8859-1')
test_vectors = vectorizer.transform(data_test.review.values.astype('U'))

lengthOfModelpre = [1 for i in range(len(data_test.review))]
# lengthOfModelpre[0] = 2
p = model.predict(test_vectors , lengths=lengthOfModelpre)

print(p)


reallabel = []
predict_texts = []


for idx in range(data_test.review.shape[0]):
    reallabel.append(data_test.sentiment[idx])

# write result into file
writer = open('./predict_result/'+str(test_path).replace(".tsv","") + ".csv", 'w')
writer.writelines("labelpredict\tlabelreal\tparagraph\n")
for index in range(0, len(data_test.review)):
    # predict_text = BeautifulSoup(data_test.review[index])
    # paragraph = str(''.join(clean_str(predict_text.get_text().encode('ascii', 'ignore'))))
    paragraph = "none"
    writer.writelines(str(p[index]) + "\t" + str(reallabel[index]) + "\t" + paragraph + "\n")
    # p[index] = 0


print(f1_score(data_test.sentiment, p, average='micro'))
print(precision_score(data_test.sentiment, p, average='micro'))
print(recall_score(data_test.sentiment, p, average='micro'))
print(accuracy_score(data_test.sentiment, p))
