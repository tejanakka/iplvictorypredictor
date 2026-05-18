from flask import Flask, render_template, request, jsonify
import pickle as pkl
import pandas as pd
import traceback

app = Flask(__name__)

# =====================
# LOAD MODELS
# =====================
teams = pkl.load(open("team.pkl", "rb"))
cities = pkl.load(open("city.pkl", "rb"))

model = pkl.load(open("pipe.pkl", "rb"))  # MUST be trained pipeline


@app.route("/")
def home():
    return render_template(
        "index.html",
        teams=sorted(teams),
        cities=sorted(cities)
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # =====================
        # INPUTS
        # =====================
        batting_team = data["batting_team"]
        bowling_team = data["bowling_team"]
        city = data["city"]

        target = int(data["target"])
        score = int(data["score"])
        overs = float(data["overs"])
        wickets_fell = int(data["wickets"])

        # =====================
        # FEATURE ENGINEERING
        # =====================
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_remaining = 10 - wickets_fell

        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # =====================
        # MODEL INPUT (IMPORTANT)
        # =====================
        input_df = pd.DataFrame([{
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "city": city,
            "runs_left": runs_left,
            "balls_left": balls_left,
            "wickets": wickets_remaining,
            "total_runs_x": target,
            "crr": crr,
            "rrr": rrr
        }])

        # =====================
        # PREDICTION
        # =====================
        result = model.predict_proba(input_df)

        win = round(result[0][1] * 100)
        loss = round(result[0][0] * 100)

        return jsonify({
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "win": win,
            "loss": loss
        })

    except Exception as e:
        print("ERROR IN /predict")
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)