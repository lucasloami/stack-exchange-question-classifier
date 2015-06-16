library('rjson')
library('tm')
library('e1071')

json_path <- './data/training.json'
test_json <- './data/testcases/input00.txt'

training_data <- fromJSON(file=json_path)
test_data <- fromJSON(file=test_json)

# df <- do.call("rbind", lapply(training_data, as.data.frame))
# df2 <- do.call("rbind", lapply(test_data, as.data.frame))
df <- lapply(training_data, function(x) {x$excerpt})
yDf <- lapply(training_data, function(x) {x$topic})
df2 <- lapply(test_data, function(x) {x$excerpt})
yDf2 <- lapply(test_data, function(x) {x$topic})


toSpace <- function(x, pattern) gsub(pattern, " ", x)
weighting <- function(x) weightTfIdf(x, normalize = FALSE)

corpus <- Corpus(VectorSource(df))
corpus <- tm_map(corpus, toSpace, "/|@|\\|")
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, PlainTextDocument)
corpus <- tm_map(corpus, stemDocument)

corpus2 <- Corpus(VectorSource(df2))
corpus2 <- tm_map(corpus2, toSpace, "/|@|\\|")
corpus2 <- tm_map(corpus2, removeNumbers)
corpus2 <- tm_map(corpus2, removePunctuation)
corpus2 <- tm_map(corpus2, removeWords, stopwords("english"))
corpus2 <- tm_map(corpus2, stripWhitespace)
corpus2 <- tm_map(corpus2, tolower)
corpus2 <- tm_map(corpus2, PlainTextDocument)

dtm <- TermDocumentMatrix(corpus, control = list(minWordLength = 1, weighting=weighting))

trainmatrix = t(dtm)
model <- naiveBayes(as.matrix(trainmatrix), as.factor(as.character(yDf)))

dtm2 <- TermDocumentMatrix(corpus2, control = list(minWordLength = 1, weighting=weighting))
testmatrix = t(dtm2)
rs<- predict(model, as.matrix(testmatrix))
