

eel.expose(my_javascript_function);
function my_javascript_function(eyes, level, orientation, inclination) {


    document.getElementById("test_id").innerHTML = eyes;
    if (eyes == "Opened"){
        document.getElementById("test_id").className = "huge success ";
    }else {
        document.getElementById("test_id").className = "huge danger";
    }

    const orientation_char = orientation[0];
    document.getElementById("orientation").src = `assets/${orientation_char}.png`;

    const level_char = level[0];
    document.getElementById("level").src = `assets/${level_char}.png`;

    document.getElementById("inclination").innerHTML = `${inclination}°`;

    document.getElementById("orientation_txt").innerHTML = orientation;
    document.getElementById("level_txt").innerHTML = level;
    const inclination_deg = parseFloat(inclination);
    if (inclination_deg < 75){
        document.getElementById("inclination_txt").innerHTML = "Danger";
        document.getElementById("inclination_txt").className = "danger text-size";

    }else {
        document.getElementById("inclination_txt").innerHTML = "Good";  
        document.getElementById("inclination_txt").className = "success text-size";

    }

    

}