from flask import Flask, render_template, request, jsonify
import pickle as pkl
import pandas as pd
import traceback

app = Flask(__name__)

# =====================
# LOAD FILES
# =====================
teams = pkl.load(open("team.pkl", "rb"))
cities = pkl.load(open("city.pkl", "rb"))

# pipeline model (IMPORTANT)
model = pkl.load(open("pipe.pkl", "rb"))

# =====================
# TEAM NAME FIX (VERY IMPORTANT)
# =====================
team_map = {
    "MI": "Mumbai Indians",
    "CSK": "Chennai Super Kings",
    "RCB": "Royal Challengers Bangalore",
    "KKR": "Kolkata Knight Riders",
    "SRH": "Sunrisers Hyderabad",
    "PBKS": "Punjab Kings",
    "KXIP": "Punjab Kings",
    "DC": "Delhi Capitals",
    "RR": "Rajasthan Royals"
}

# =====================
# HOME ROUTE
# =====================
@app.route("/")
def home():
    return render_template(
        "index.html",
        teams=sorted(teams),
        cities=sorted(cities)
    )

# =====================
# PREDICT ROUTE
# =====================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # raw input
        batting_team = data["batting_team"]
        bowling_team = data["bowling_team"]
        city = data["city"]

        # convert abbreviations → full names
        batting_team = team_map.get(batting_team, batting_team)
        bowling_team = team_map.get(bowling_team, bowling_team)

        target = int(data["target"])
        score = int(data["score"])
        overs = float(data["overs"])
        wickets_fell = int(data["wickets"])

        # feature engineering
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_remaining = 10 - wickets_fell

        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # input dataframe (MUST MATCH TRAINING)
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

        # prediction
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
        print("❌ ERROR IN /predict")
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


# =====================
# RUN APP
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)