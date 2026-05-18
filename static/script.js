async function predict(){

    const btn = document.querySelector(".predict-btn");

    btn.innerHTML =
    `<span class="spinner-border spinner-border-sm">
    </span> Predicting...`;

    btn.disabled = true;

    const data = {
        batting_team:
        document.getElementById("batting_team").value,

        bowling_team:
        document.getElementById("bowling_team").value,

        city:
        document.getElementById("city").value,

        target:
        document.getElementById("target").value,

        score:
        document.getElementById("score").value,

        overs:
        document.getElementById("overs").value,

        wickets:
        document.getElementById("wickets").value
    };

    try{

        const response = await fetch("/predict",{
            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify(data)
        });

        const result = await response.json();

        const resultBox =
        document.getElementById("resultBox");

        resultBox.classList.remove("d-none");

        document.getElementById(
        "battingText").innerText =
        result.batting_team;

        document.getElementById(
        "battingPercent").innerText =
        result.win + "%";

        document.getElementById(
        "bowlingText").innerText =
        result.bowling_team;

        document.getElementById(
        "bowlingPercent").innerText =
        result.loss + "%";

        setTimeout(()=>{

            document.getElementById(
            "battingBar").style.width =
            result.win + "%";

            document.getElementById(
            "bowlingBar").style.width =
            result.loss + "%";

        },200);

    }

    catch(error){
        alert("Prediction failed");
        console.log(error);
    }

    btn.innerHTML =
    `<i class="fas fa-chart-line"></i>
    Predict Probabilities`;

    btn.disabled = false;
}