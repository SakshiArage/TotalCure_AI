from waitress import serve
import main  # Import your Flask app

if __name__ == '__main__':
    print("Starting production server on http://127.0.0.1:5001")
    serve(main.app, host='127.0.0.1', port=5001, threads=4, url_scheme='http') 