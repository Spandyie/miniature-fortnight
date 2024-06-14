from flask import Flask, render_template, request, redirect, url_for
import tweepy as tw
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
import traceback
import csv
import torch
import torch.nn as nn
from collections import Counter
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ProjectDis.db'

db = SQLAlchemy(app)

class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_size, batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)
        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size), torch.zeros(1, batch_size, self.lstm_size))

def get_data_from_file(train_file, batch_size, seq_size):
        with open(train_file, 'r') as f:
            text = f.read()
        text = text.split()

        word_counts = Counter(text)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
        vocab_to_int = {w: k for k, w in int_to_vocab.items()}
        n_vocab = len(int_to_vocab)

        print('Vocabulary size', n_vocab)

        int_text = [vocab_to_int[w] for w in text]
        num_batches = int(len(int_text) / (seq_size * batch_size))
        in_text = int_text[:num_batches * batch_size * seq_size]
        out_text = np.zeros_like(in_text)
        out_text[:-1] = in_text[1:]
        out_text[-1] = in_text[0]
        in_text = np.reshape(in_text, (batch_size, -1))
        out_text = np.reshape(out_text, (batch_size, -1))
        return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text


def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k):
    net.eval()
    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])
    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    return  ' '.join(words)


class ProjectDatabase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    searchQuery = db.Column(db.String(20), nullable=False)
    username = db.Column(db.String(20), nullable=False)
    tweet = db.Column(db.String(280), nullable=False)
    location = db.Column(db.String(20), nullable=True)

    def __repr__(self):
        return '<User {0}>'.format(self.id)


@app.route('/', methods=['POST', 'GET'])
def my_input():
     return render_template("input.html")


@app.route('/displayResult', methods=['POST'])
def displayResult():
    N=5
    search_word = request.form['hashtag']
    consumer_key = ....
    consumer_secret_key= ....
    access_token = ...
    access_token_key = ...
    # setup authorization
    authorization_handle = tw.OAuthHandler(consumer_key=consumer_key, consumer_secret=consumer_secret_key)
    authorization_handle.set_access_token(key=access_token, secret=access_token_key)
    api = tw.API(authorization_handle,wait_on_rate_limit=True)
    # search for the keyword
    number_of_days_ago = datetime.now() - timedelta(N)
    since_date = number_of_days_ago.date().strftime("%Y-%m-%d")
    try:
        updated_search_word = search_word + "-filter:retweets"
        tweet_object = tw.Cursor(api.search, q=updated_search_word, lang='en',since= since_date).items(50)
        tweet_array = [{"username": tweet.user.screen_name,"tweet": tweet.text,"location":tweet.user.location} for tweet in tweet_object]
        try:
            for tweetItr in tweet_array:
                task = ProjectDatabase(searchQuery=search_word,username=tweetItr['username'], tweet=tweetItr['tweet'], location=tweetItr['location'])
                db.session.add(task)
                db.session.commit()
                #print("Done inseting")
        except:
            traceback.print_exc()

    except:
        print("Error")

    return render_template("display.html", tweet_array=tweet_array)

@app.route('/trump', methods=['POST', 'GET'])
def mineTrump():
    N= 5
    consumer_key = ...
    consumer_secret_key = ...
    access_token = ...
    access_token_key = ...
    # setup authorization
    authorization_handle = tw.OAuthHandler(consumer_key=consumer_key, consumer_secret=consumer_secret_key)
    authorization_handle.set_access_token(key=access_token, secret=access_token_key)
    api = tw.API(authorization_handle, wait_on_rate_limit=True)
    # search for the keyword
    number_of_days_ago = datetime.now() - timedelta(N)
    since_date = number_of_days_ago.date().strftime("%Y-%m-%d")
    all_trump_tweets = list()
    # make request for the most recent 200 tweets
    new_tweets = api.user_timeline(screen_name='@realDonaldTrump',count=200)
    # save the id of oldest tweet -1

    all_trump_tweets.extend(new_tweets)
    oldest = all_trump_tweets[-1].id-1
    # keeo grabbing the new tweets
    while(len(new_tweets) > 0):
        new_tweets=api.user_timeline(screen_name='@realDonaldTrump',count=200,max_id=oldest)
        # save the most recent tweets
        all_trump_tweets.extend(new_tweets)
        #update the id of the oldest tweet less one
        oldest= all_trump_tweets[-1].id-1
    final_parsed_tweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in all_trump_tweets]

    with open("trumpTweets.csv", 'a+') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "created_at", "text"])
        writer.writerows(final_parsed_tweets)

    return render_template("displayTrump.html", final_parsed_tweets=final_parsed_tweets)

@app.route('/tweetliketrump', methods=['POST', 'GET'])
def predictTrump():
    words = [request.form["keyword"]]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file('Train.txt', 16, 10)
    model = RNNModule(n_vocab, 10, 64, 64)
    model.load_state_dict(torch.load('model-5000.pth'))
    predicted_tweet = predict(device, model, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5)
    return render_template("tweetLikeTrump.html", synthetic_tweet=predicted_tweet)

@app.route('/getTrumped', methods=['GET'])
def getTrumped():
    return render_template('enterTrumpTweet.html')

if __name__ == '__main__':
    app.run(debug=True)
