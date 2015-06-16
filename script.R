library('rjson')
library('tm')
library('e1071')

json_path <- './data/training.json'
test_json <- './data/testcases/input00.txt'

training_data <- fromJSON(file=json_path)
test_data <- fromJSON(file=test_json)

df <- do.call("rbind", lapply(training_data, as.data.frame))
df2 <- do.call("rbind", lapply(test_data, as.data.frame))

toSpace <- function(x, pattern) gsub(pattern, " ", x)
weighting <- function(x) weightTfIdf(x, normalize = FALSE)

corpus <- Corpus(VectorSource(df$excerpt))
corpus <- tm_map(corpus, toSpace, "/|@|\\|")
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, PlainTextDocument)
corpus <- tm_map(corpus, stemDocument)
# print(as.character(corpus[[1]]))


dtm <- TermDocumentMatrix(corpus, control = list(minWordLength = 1, weighting=weighting))

trainmatrix = t(dtm)
model <- naiveBayes(as.matrix(trainmatrix),as.factor(df$topic))
print(model)

