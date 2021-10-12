const realFileBtn = document.getElementsByClassName("file_gambar");
const customBtn = document.getElementById("custom-button");
const customTxt = document.getElementById("custom-text");



customBtn.addEventListener("click", function() {
  realFileBtn.click();
});

realFileBtn.addEventListener("change", function() {
  if (realFileBtn.value) {
    customTxt.innerHTML = realFileBtn.value.match()[1];
  } else {
    customTxt.innerHTML = "No file chosen, yet.";
  }
});


