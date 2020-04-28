rm (list=ls())

# Methods Implemented
# Doc-Term Matrix,Neural Network,Removing stop words, tfdif, tokenization, pruning
# Models
# Random Forest, GBM, GLM(Don't know if using this make sense. But used the converted numbers), Neural Network
# Cross Validation
# For unknown class attempt to make look at words that are not in train data but were in test

library(text2vec)
library(keras)
library(tensorflow)
library(caret)
library(h2o)


train<-read.csv("train.csv")
test<-read.csv("test.csv")

train$NARRATIVE<-as.character(train$NARRATIVE)  # changes train to char
test$NARRATIVE<-as.character(test$NARRATIVE)  # changes test to char

train$TextLength <- nchar(train$NARRATIVE)   # adds Textlength
test$TextLength <- nchar(test$NARRATIVE)


library(tm)
crimeCorpus= Corpus(VectorSource(train$NARRATIVE))
crimeCorpus_Test=Corpus(VectorSource(test$NARRATIVE))
print(crimeCorpus)
inspect(crimeCorpus[1:3])
inspect(crimeCorpus_Test[1:3])

#translate to lower case
crimeCorpusClean=tm_map(crimeCorpus,tolower)
crimeCorpusClean_Test=tm_map(crimeCorpus_Test,tolower)

print(crimeCorpusClean)
inspect(crimeCorpusClean[1:3])
inspect(crimeCorpusClean_Test[1:3])



#Remove numbers
crimeCorpusClean=tm_map(crimeCorpusClean,removeNumbers)
crimeCorpusClean_Test=tm_map(crimeCorpusClean_Test,removeNumbers)
#Remove punctuation
crimeCorpusClean=tm_map(crimeCorpusClean,removePunctuation)
crimeCorpusClean_Test=tm_map(crimeCorpusClean_Test,removePunctuation)
#Strip whitespace
crimeCorpusClean=tm_map(crimeCorpusClean,stripWhitespace)
crimeCorpusClean_Test



#Inspect the clean corpus
inspect(crimeCorpusClean[1:3])
inspect(crimeCorpusClean_Test[1:3])

#Creating document term matrix for tokenization of corpus
crime_dtm=DocumentTermMatrix(crimeCorpusClean)


# bro=tokens_select(crime_dtm, stopwords(), 
#               selection = "remove")

inspect(crime_dtm[1:10, 10:15])

crime_dtm=DocumentTermMatrix(crimeCorpusClean,control=list(
  tolower = TRUE,
  removeNumbers=TRUE,
  removeWords=TRUE,
  removePunctuation=TRUE,
  stripWhitespace=TRUE
  
))

crime_dtm_test=DocumentTermMatrix(crimeCorpus_Test,control=list(
  tolower = TRUE,
  removeNumbers=TRUE,
  removeWords=TRUE,
  removePunctuation=TRUE,
  stripWhitespace=TRUE
  
))

inspect(crime_dtm_test[1:10, 10:15])

 ############################################Class VIDEO#########################################################
use_condaenv("r-tensorflow")
train$NARRATIVE<-as.character(train$NARRATIVE)
txt=as.character(train$NARRATIVE)
it= itoken(txt,tolower,word_tokenizer,n_chunks = 10)
vocab=create_vocabulary(it)
vocab=prune_vocabulary(vocab,term_count_min = 100,
                       doc_proportion_max = 0.25)
word_vectorizer=vocab_vectorizer(vocab)


it_train= itoken(train$NARRATIVE,
                 preprocessor = tolower,
                 tokenizer = word_tokenizer,
                 ids=train$DR,
                 progressbar = FALSE)
dtm_train=create_dtm(it_train,word_vectorizer)






label_fac=as.numeric(as.factor(train$CRIMETYPE))
label_fac=label_fac-1
ytrain=label_fac


y_train = to_categorical(ytrain)

model=keras_model_sequential()
model %>%
  layer_dense()




train=as.matrix(dtm_train)
history<-model %>%fir(
  train,y_train,
  epochs=10,batch_size=100,
  validation_split=0.2)
)


######
library(gbm)
crime_gbm=gbm(CRIMETYPE~NARRATIVE,
             data=train,distribution="bernoulli",
             interaction.depth = 3,n.trees=100)


library(glm)
crime_glm=glm(CRIMETYPE~NARRATIVE,
              data=train,distribution="bernoulli",
              interaction.depth = 3,n.trees=100)

model_glm=glm(CRIMETYPE~NARRATIVE,
              train,
              family="binomial")
####



#finding the frequent terms
frequent_words=findFreqTerms(crime_dtm,10)   #TRAIN
length(frequent_words)

frequent_words_test=findFreqTerms(crime_dtm_test,10)
length(frequent_words_test)



# View some of the frequent words
frequent_words[1:10]
frequent_words_test[1:10]



#Creating document term matrix with frequent words
crime_freq_word_train=crime_dtm[,frequent_words]
crime_freq_word_test=crime_dtm_test[,frequent_words_test]  #TEst


##YES/NO
yes_or_no=function(x){
  y=ifelse(x>0,1,0)
  y=factor(y,levels=c(1,0),labels = c("No","Yes"))
  y
}

crime_train=apply(crime_freq_word_train,2,yes_or_no)
crime_train2=apply(crime_freq_word_train,2,yes_or_no)

crime_test=apply(crime_freq_word_test,2,yes_or_no)


library(e1071)
crime_classifier=naiveBayes(crime_train,train$CRIMETYPE)
crime_classifier2=naiveBayes(crime_freq_word_train,train$CRIMETYPE)

class(crime_classifier)

crime_test_pred=predict(crime_classifier,newdata=crime_test)


##########################################YOUTBE TEXT ANALYTICS ####################################################
library(ggplot2)


prop.table(table(train$CRIMETYPE)) #Distribution of Crimetype labels

# 
train$NARRATIVE<-as.character(train$NARRATIVE)  # changes to char
test$NARRATIVE<-as.character(test$NARRATIVE)  # changes test to char


train$TextLength <- nchar(train$NARRATIVE)
summary(train$TextLength)

ggplot(train, aes(x = TextLength, fill = CRIMETYPE)) +
  theme_bw() +
  geom_histogram(binwidth = 5) +
  labs(y = "Text Count", x = "Length of Text",
       title = "Distribution of Text Lengths with Class Labels")

#Tokenization
library(quanteda)
# Tokenize SMS text messages.
train.tokens <- tokens(train$NARRATIVE, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)

train.tokens[[20]]

# Lowercasing the tokens
train.tokens <- tokens_tolower(train.tokens)


# Use quanteda's built-in stopword list for English.

train.tokens <- tokens_select(train.tokens, stopwords(), 
                              selection = "remove")

# Perform stemming on the tokens.
train.tokens <- tokens_wordstem(train.tokens, language = "english")


# Create our first bag-of-words model.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)


# Transform to a matrix and inspect.
train.tokens.matrix <- as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20, 1:100])
dim(train.tokens.matrix)
dim(train.tokens.dfm)


# Setup a the feature data frame with labels.
#train.tokens.df <- cbind(CRIMETYPE = train$CRIMETYPE, as.data.frame(train.tokens.dfm))
train.tokens.df <- cbind(CRIMETYPE = train$CRIMETYPE, convert(train.tokens.dfm,to="data.frame",docvars = NULL))


# Cleanup column names.
names(train.tokens.df) <- make.names(names(train.tokens.df),unique=TRUE)

# #3 times (i.e., create 30 random stratified samples)
# set.seed(48743)
# cv.folds <- createMultiFolds(train$CRIMETYPE, k = 10, times = 1)
# cv.cntrl <- trainControl(method = "repeatedcv", number = 3,
#                          repeats = 1, index = cv.folds)
# 
# ###CV
# #install.packages("doSNOW")
# library(doSNOW)
# 
# 
# # Time the code execution
# start.time <- Sys.time()
# 
# 
# # Create a cluster to work on 10 logical cores.
# cl <- makeCluster(3, type = "SOCK")
# registerDoSNOW(cl)
# 
# 
# # As our data is non-trivial in size at this point, use a single decision
# # tree alogrithm as our first model. We will graduate to using more
# # powerful algorithms later when we perform feature extraction to shrink
# # the size of our data.
# rpart.cv.1 <- train(CRIMETYPE ~ ., train.tokens.df, method = "rpart")
# 
# 
# # Processing is done, stop cluster.
# stopCluster(cl)
# 
# 
# # Total time of execution on workstation was approximately 4 minutes.
# total.time <- Sys.time() - start.time
# total.time
# 
# 
# # Check out our results.
# rpart.cv.1


####################################3Term Frequency-Inverse Document Frequency (TF-IDF) 
# Our function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}

# Our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}


# Our function for calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}

# First step, normalize all documents via TF.
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)


# Second step, calculate the IDF vector that we will use - both
# for training data and for test data!
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)

# Lastly, calculate TF-IDF for our training corpus.
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])

# Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])






###############################CLASS EXAMPLE#################################################################
library(text2vec)
library(keras)
library(tensorflow)
library(caret)


train<-read.csv("train.csv")
test<-read.csv("test.csv")
use_condaenv("r-tensorflow")
################train####################3
train$NARRATIVE<-as.character(train$NARRATIVE)
txt=as.character(train$NARRATIVE)
it= itoken(txt,tolower,word_tokenizer,n_chunks = 10)
vocab=create_vocabulary(it)
vocab=prune_vocabulary(vocab,term_count_min = 100,
                       doc_proportion_max = 0.25)
word_vectorizer=vocab_vectorizer(vocab)


it_train= itoken(train$NARRATIVE,
                 preprocessor = tolower,
                 tokenizer = word_tokenizer,
                 ids=train$DR,
                 progressbar = FALSE)
dtm_train=create_dtm(it_train,word_vectorizer)


label_fac=as.numeric(as.factor(train$CRIMETYPE))
label_fac=label_fac-1
ytrain=label_fac


y_train = to_categorical(ytrain)

model=keras_model_sequential()
model %>%
  layer_dense(units=20,activation='relu',input_shape=c(dim(dtm_train)[2])) %>%
  layer_dropout(rate=0.2)%>%
  layer_dense(units=length(unique(ytrain)),activation='softmax')

model %>% compile(
  loss='categorical_crossentropy',
  optimizer=optimizer_rmsprop(),
  metrics=c('accuracy')
)

trainM=as.matrix(dtm_train)
history<- model %>% fit(
  trainM,y_train,
  epochs=10,batch_size=100,
  validation_split=0.2
)

model %>% evaluate(trainM,y_train)
preds<-predict_classes(object=model,x=test)%>% as.vector()

###########TEST###########################################
test$NARRATIVE<-as.character(test$NARRATIVE)
txtTEST=as.character(test$NARRATIVE)
itTEST= itoken(txtTEST,tolower,word_tokenizer,n_chunks = 10)
vocabTEST=create_vocabulary(itTEST)
vocabTEST=prune_vocabulary(vocabTEST,term_count_min = 100,
                           doc_proportion_max = 0.25)
word_vectorizerTEST=vocab_vectorizer(vocabTEST)


it_TEST= itoken(test$NARRATIVE,
                preprocessor = tolower,
                tokenizer = word_tokenizer,
                ids=test$DR,
                progressbar = FALSE)
dtm_TEST=create_dtm(it_TEST,word_vectorizer)



testM=as.matrix(dtm_TEST)


preds<-predict_classes(object=model,x=testM)%>% as.vector()
test$CRIMETYPE<-preds
back<-as.factor(as.numeric(preds))
submission=test[,c("DR","CRIMETYPE")]
as.label(preds, prefix=TRUE)
write.csv(submission,"sub1.csv",row.names=F)

# Transform to dfm and then a matrix.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
train.tokens.dfm



################################################
library("kernlab") 
library("caret") 
library("tm") 
library("dplyr") 
library("splitstackshape")
library("e1071")
train <- tm_map(train, content_transformer(stripWhitespace))
train <- tm_map(train, content_transformer(tolower))
train <- tm_map(train, content_transformer(removeNumbers))
train <- tm_map(train, content_transformer(removePunctuation))


df.model<-ksvm(CRIMETYPE~., data= train, kernel="rbfdot")