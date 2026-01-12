document.addEventListener("DOMContentLoaded", function () {
  const images = [
    { src: "/static/images/nkar1.jpeg", style: { top: "400px", left: "600px" } },
    { src: "/static/images/nkar2.png", style: { top: "400px", left: "600px" } },
    { src: "/static/images/nkar3.jpg", style: { bottom: "20px", right: "800px" } },
    { src: "/static/images/nkar4.jpg", style: { top: "50%", left: "50%", transform: "translate(-50%, -50%)" } }
  ];

  let index = 0;
  const slide = document.getElementById("slide");
  const caption = document.getElementById("caption");

  function changeImage() {
    index++;
    if (index >= images.length) index = 0;

    slide.src = images[index].src;
    caption.textContent = images[index].text;

    // reset style
    caption.style.top = "";
    caption.style.bottom = "";
    caption.style.left = "";
    caption.style.right = "";
    caption.style.transform = "";

    Object.assign(caption.style, images[index].style);
  }

  // run first time
  changeImage();

  // run every 3 sec
  setInterval(changeImage, 3000);
});