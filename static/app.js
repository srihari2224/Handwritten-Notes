const canvas = document.getElementById("pad");
const ctx = canvas.getContext("2d");
const calcBtn = document.getElementById("calc");
const clearBtn = document.getElementById("clear");
const brushInput = document.getElementById("brush");
const exprEl = document.getElementById("expr");
const resultEl = document.getElementById("result");
const predsEl = document.getElementById("preds");
const errorEl = document.getElementById("error");
const annotImg = document.getElementById("annot");

// Setup white background
function resetCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
resetCanvas();

let drawing = false;
let lastX = 0;
let lastY = 0;

function getPos(evt) {
  const rect = canvas.getBoundingClientRect();
  const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
  const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
  return {
    x: (clientX - rect.left) * (canvas.width / rect.width),
    y: (clientY - rect.top) * (canvas.height / rect.height),
  };
}

function startDraw(evt) {
  drawing = true;
  const p = getPos(evt);
  lastX = p.x;
  lastY = p.y;
}

function draw(evt) {
  if (!drawing) return;
  const size = parseInt(brushInput.value, 10) || 25;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = size;
  const p = getPos(evt);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  lastX = p.x;
  lastY = p.y;
}

function endDraw() { drawing = false; }

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", endDraw);
canvas.addEventListener("touchstart", (e) => { e.preventDefault(); startDraw(e); });
canvas.addEventListener("touchmove", (e) => { e.preventDefault(); draw(e); });
canvas.addEventListener("touchend", (e) => { e.preventDefault(); endDraw(); });

clearBtn.addEventListener("click", () => {
  resetCanvas();
  exprEl.textContent = "—";
  resultEl.textContent = "—";
  predsEl.innerHTML = "";
  errorEl.textContent = "";
  annotImg.removeAttribute("src");
});

calcBtn.addEventListener("click", async () => {
  // Convert to DataURL (PNG)
  const dataURL = canvas.toDataURL("image/png");
  exprEl.textContent = "Processing...";
  resultEl.textContent = "—";
  predsEl.innerHTML = "";
  errorEl.textContent = "";
  annotImg.removeAttribute("src");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataURL }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      errorEl.textContent = data.error || "Server error";
      exprEl.textContent = "—";
      return;
    }
    exprEl.textContent = data.expression ?? "—";
    resultEl.textContent = data.result ?? "—";
    if (data.annotated_image) annotImg.src = data.annotated_image;

    predsEl.innerHTML = "";
    (data.tokens || []).forEach(t => {
      const li = document.createElement("li");
      li.textContent = `${t.symbol} — ${t.confidence}%`;
      predsEl.appendChild(li);
    });
  } catch (e) {
    errorEl.textContent = String(e);
    exprEl.textContent = "—";
  }
});
