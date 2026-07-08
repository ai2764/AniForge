(() => {
  const form = document.getElementById("generate-form");
  const imageInput = document.getElementById("image-input");
  const dropZone = document.getElementById("drop-zone");
  const dropZoneText = document.getElementById("drop-zone-text");
  const dropZonePreview = document.getElementById("drop-zone-preview");
  const statusEl = document.getElementById("status");
  const errorsEl = document.getElementById("errors");
  const preview = document.getElementById("preview");
  const idleVideo = document.getElementById("idleVideo");
  const actionVideo = document.getElementById("actionVideo");

  let hasAction = false;

  // -- image picker (click or drag) --------------------------------------
  dropZone.addEventListener("click", () => imageInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) {
      imageInput.files = e.dataTransfer.files;
      showImagePreview(file);
    }
  });

  imageInput.addEventListener("change", () => {
    const file = imageInput.files && imageInput.files[0];
    if (file) showImagePreview(file);
  });

  function showImagePreview(file) {
    const url = URL.createObjectURL(file);
    dropZonePreview.src = url;
    dropZonePreview.style.display = "block";
    dropZoneText.textContent = file.name;
    dropZone.classList.add("has-image");
  }

  // -- submit --------------------------------------------------------------
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorsEl.textContent = "";

    if (!imageInput.files || !imageInput.files[0]) {
      statusEl.textContent = "Please choose an image.";
      return;
    }

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);
    formData.append("action_prompt", document.getElementById("action_prompt").value);
    formData.append("idle_prompt", document.getElementById("idle_prompt").value);
    for (const box of form.querySelectorAll('input[name="overshoot"]:checked')) {
      formData.append("overshoot", box.value);
    }
    const seedValue = document.getElementById("seed").value.trim();
    if (seedValue !== "") {
      formData.append("seed", seedValue);
    }

    statusEl.textContent = "Generating...";
    hasAction = false;
    preview.classList.remove("has-action");
    actionVideo.pause();
    actionVideo.style.display = "none";

    try {
      const res = await fetch("/generate", { method: "POST", body: formData });
      const data = await res.json();

      if (data.idle) {
        idleVideo.src = data.idle;
        idleVideo.style.display = "block";
        idleVideo.play();
      }

      if (data.action) {
        actionVideo.src = data.action;
        hasAction = true;
        preview.classList.add("has-action");
      }

      if (data.idle || data.action) {
        preview.style.display = "block";
      }

      if (data.errors && Object.keys(data.errors).length > 0) {
        errorsEl.textContent = JSON.stringify(data.errors, null, 2);
      }

      if (data.seed !== undefined && data.seed !== null) {
        document.getElementById("seed").value = data.seed;
        statusEl.textContent = "Done. (seed " + data.seed + ")";
      } else {
        statusEl.textContent = "Done.";
      }
    } catch (err) {
      statusEl.textContent = "Request failed.";
      errorsEl.textContent = String(err);
    }
  });

  // -- click-triggered player ----------------------------------------------
  preview.addEventListener("click", () => {
    if (!hasAction) return;
    idleVideo.pause();
    idleVideo.style.display = "none";
    actionVideo.style.display = "block";
    actionVideo.currentTime = 0;
    actionVideo.play();
  });

  actionVideo.addEventListener("ended", () => {
    actionVideo.style.display = "none";
    idleVideo.style.display = "block";
    idleVideo.play();
  });
})();
