from app.init_flask_app import init

app = init()

if __name__ == '__main__':
    app.run(debug=True)