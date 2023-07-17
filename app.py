from flask import Flask, render_template, request
from main import model

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route("/", methods=['POST'])
def get_recommend():
    userid = int(request.form['userid'])

    movie_user, recommendor_cont, recommendor_colab, recommendor_hybrid = model(
        userid)

    return render_template('home.html', userid=userid, movie_user=movie_user, recommendor_cont=recommendor_cont,
                           recommendor_colab=recommendor_colab, recommendor_hybrid=recommendor_hybrid)


if __name__ == '__main__':
    app.run("localhost", "9999", debug=True)
