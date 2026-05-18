from flask import Flask, render_template, request, jsonify
import pickle as pkl
import pandas as pd
import traceback

app = Flask(__name__)

# =====================
# LOAD MODEL
# =====================
teams = pkl.load(open("team.pkl", "rb"))
cities = pkl.load(open("city.pkl", "rb"))

model = pkl.load(open("pipe.pkl", "rb"))

# =====================
# HOME
# =====================
@app.route("/")
def home():
    return render_template(
        "index.html",
        teams=sorted(teams),
        cities=sorted(cities)
    )

# =====================
# PREDICT
# =====================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        batting_team = data["batting_team"]
        bowling_team = data["bowling_team"]
        city = data["city"]

        target = int(data["target"])
        score = int(data["score"])
        overs = float(data["overs"])
        wickets_fell = int(data["wickets"])

        # Feature engineering
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_remaining = 10 - wickets_fell

        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # IMPORTANT: must match training columns
        input_df = pd.DataFrame([{
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "city": city,
            "Score": score,
            "Wickets": wickets_remaining,
            "Remaining Balls": balls_left,
            "target_left": runs_left,
            "crr": crr,
            "rrr": rrr
        }])

        result = model.predict_proba(input_df)

        return jsonify({
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "win": round(result[0][1] * 100),
            "loss": round(result[0][0] * 100)
        })

    except Exception as e:
        print("ERROR:")
        traceback.print_exc()

        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)