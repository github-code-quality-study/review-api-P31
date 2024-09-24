import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        self.allowed_locations = [
            "Albuquerque, New Mexico",
            "Carlsbad, California",
            "Chula Vista, California",
            "Colorado Springs, Colorado",
            "Denver, Colorado",
            "El Cajon, California",
            "El Paso, Texas",
            "Escondido, California",
            "Fresno, California",
            "La Mesa, California",
            "Las Vegas, Nevada",
            "Los Angeles, California",
            "Oceanside, California",
            "Phoenix, Arizona",
            "Sacramento, California",
            "Salt Lake City, Utah",
            "San Diego, California",
            "Tucson, Arizona"
        ]
        
    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Parse query parameters
            query_params = parse_qs(environ.get("QUERY_STRING", ""))
            location = query_params.get("location", [""])[0]
            start_date_str = query_params.get("start_date", [""])[0]
            end_date_str = query_params.get("end_date", [""])[0]

            # Convert date strings to datetime objects
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None

            filtered_reviews = reviews

            # Filter reviews by location
            if location and location in self.allowed_locations:
                filtered_reviews = [review for review in reviews if review.get("Location") == location]

            # Filter reviews by date range
            if start_date:
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review.get("Timestamp"), "%Y-%m-%d %H:%M:%S") >= start_date]
            if end_date:
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review.get("Timestamp"), "%Y-%m-%d %H:%M:%S") <= end_date]


            # add sentiment analysis to each review
            for review in filtered_reviews:
                review["sentiment"] = self.analyze_sentiment(review["ReviewBody"])

            # filter in descending order by the compound value in sentiment.
            filtered_reviews = sorted(filtered_reviews, key=lambda x: x["sentiment"]["compound"], reverse=True)

            # Create the response body from the filtered reviews and convert to a JSON byte string
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                request_body_size = int(environ.get("CONTENT_LENGTH", 0))
                request_body = environ["wsgi.input"].read(request_body_size)
                input_data = parse_qs(request_body.decode("utf-8"))

                review_body = input_data.get("ReviewBody", [""])[0]
                location = input_data.get("Location", [""])[0]

                if not review_body:
                    response_body = json.dumps({"error": "Review body is required"}).encode("utf-8")
                    start_response("400 Bad Request", [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(response_body)))
                    ])
                    return [response_body]

                if location not in self.allowed_locations:
                    response_body = json.dumps({"error": "Invalid location"}).encode("utf-8")
                    start_response("400 Bad Request", [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(response_body)))
                    ])
                    return [response_body]

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                review_id = str(uuid.uuid4())

                new_review = {
                    "ReviewId": review_id,
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": timestamp
                }

                reviews.append(new_review)

                response_body = json.dumps(new_review).encode("utf-8")
                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]
            except Exception as e:
                response_body = json.dumps({"error": str(e)}).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]
                            
if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()