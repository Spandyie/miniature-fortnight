from flask import Flask, render_template, request, redirect, url_for
import tweepy as tw
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = ' sqlite:///project.db'

db = SQLAlchemy(app)

class ProjectDatabase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(10),nullable=False)
    tweet = db.Column(db.String, nullable=False)
    location = db.Column(db.String, nullable=False)

    def __repr__(self):
        return '<Data {0}>'.format(self.id)


@app.route('/', methods=['POST','GET'])
def my_input():
     return render_template("input.html")


@app.route('/displayResult', methods=['POST'])
def displayResult():
    N=5
    search_word = request.form['hashtag']
    consumer_key = 'jGmHZTqngDM6j10ShQUssWwk3'
    consumer_secret_key= 'bdpCgUeii0HITOClnd6cqhKCEXwKARG5HvtCMBafGlXP0fUah3'
    access_token = '97490655-XsIlPX4rsJ0mMWUtNUjERKUoGuH6WDZlz3lU1jDu0'
    access_token_key = 'gZDAsgaEk9g5nXfyi18GagAA4G2CDafSGG8ikgLcmweKc'
    # setup authorization
    authorization_handle = tw.OAuthHandler(consumer_key=consumer_key, consumer_secret=consumer_secret_key)
    authorization_handle.set_access_token(key=access_token, secret=access_token_key)
    api = tw.API(authorization_handle,wait_on_rate_limit=True)
    # search for the keyword
    number_of_days_ago = datetime.now() - timedelta(N)
    since_date = number_of_days_ago.date().strftime("%Y-%m-%d")
    try:
        search_word = search_word + "-filter:retweets"
        tweet_object = tw.Cursor(api.search, q=search_word, lang='en',since= since_date).items(50)
        tweet_array = [{"username": tweet.user.screen_name,"tweet": tweet.text,"location":tweet.user.location} for tweet in tweet_object]
        # create pandas dataframe
        #tweet_data_frame = pd.DataFrame(tweet_array, columns=['user_name', 'tweet', 'locations'])
        #print(tweet_data_frame)

    except:
        print("Error")

    return render_template("display.html", tweet_array=tweet_array)



if __name__ == '__main__':
    app.run(debug=True)
