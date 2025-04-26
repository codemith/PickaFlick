
# PickaFlick: Personalized Movie Recommendation System

**PickaFlick** is a personalized movie recommendation system that suggests movies based on your input using collaborative filtering. It features an interactive web interface built with Streamlit and integrates with the TMDB API to display movie posters for an enhanced user experience.

---

## Setup Instructions

### 1. Clone the Repository
```
git clone https://github.com/codemith/PickaFlick.git
cd PickaFlick
```

### 2. Set Up Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download MovieLens Dataset
- Download the **MovieLens 1M** dataset from [GroupLens](https://grouplens.org/datasets/movielens/1m/)
- Extract and place it in:
```
PickaFlick/data/raw/ml-1m/
```

---

## Train the Recommendation Model
Run the training script to preprocess data and train the model:
```
python train.py
```
This will generate:
- `models/mf_model_weights.pth`
- `models/user_encoder.pkl`
- `models/movie_encoder.pkl`

---

## Run the Streamlit App Locally
1. Create a `.env` file in the root directory containing:
```
TMDB_API_KEY=your_tmdb_api_key_here
```

2. Launch the app:
```
streamlit run recommend_app.py
```

3. Open your browser and navigate to:
```
http://localhost:8501
```


## Deploying on AWS EC2
- SSH into your EC2 instance.
- Clone this repository and repeat the setup steps.
- Run:
```
streamlit run recommend_app.py --server.port 8501 --server.address 0.0.0.0
```
- Access via:
```
http://<Your-EC2-Public-IP>:8501
```

For persistent hosting, configure **systemd** 

## How It Works
- Enter a movie title in the search bar.
- Use the slider to select the number of recommendations.
- View similar movies along with posters and genres fetched via TMDB API.

