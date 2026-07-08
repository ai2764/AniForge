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

  const POSE_HINTS = {
    standing: "Standing: free motion from prompts (no pose pin).",
    sitting: "Sitting: pin pelvis + feet from the image; upper body free.",
    lying: "Lying: full skeleton lock to the image pose (small motion only).",
  };
  const poseHint = document.getElementById("pose-hint");
  const jointOvershoot = document.getElementById("overshoot-joint");

  function selectedPoseMode() {
    const el = form.querySelector('input[name="pose_mode"]:checked');
    return el ? el.value : "standing";
  }

  function syncPoseUi() {
    const mode = selectedPoseMode();
    if (poseHint) poseHint.textContent = POSE_HINTS[mode] || POSE_HINTS.standing;
    // Joint overshoot is SOMA/standing-path only.
    if (jointOvershoot) {
      const standing = mode === "standing";
      jointOvershoot.disabled = !standing;
      if (!standing) jointOvershoot.checked = false;
    }
  }

  form.querySelectorAll('input[name="pose_mode"]').forEach((el) => {
    el.addEventListener("change", syncPoseUi);
  });
  syncPoseUi();

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
    formData.append("pose_mode", selectedPoseMode());
    for (const box of form.querySelectorAll('input[name="overshoot"]:checked')) {
      formData.append("overshoot", box.value);
    }
    const seedValue = document.getElementById("seed").value.trim();
    if (seedValue !== "") {
      formData.append("seed", seedValue);
    }

    const mode = selectedPoseMode();
    statusEl.textContent =
      mode === "standing" ? "Generating..." : "Generating (" + mode + ", pose-anchored)...";
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
        actionVideo.load();
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
  function returnToIdle() {
    actionVideo.pause();
    actionVideo.style.display = "none";
    idleVideo.style.display = "block";
    idleVideo.play().catch(() => {});
  }

  preview.addEventListener("click", () => {
    if (!hasAction) return;
    if (actionVideo.error) {
      errorsEl.textContent = "Action video failed to load (unsupported or missing).";
      return;
    }
    idleVideo.pause();
    idleVideo.style.display = "none";
    actionVideo.style.display = "block";
    try {
      actionVideo.currentTime = 0;
    } catch (_) {
      /* ignore if metadata not ready yet */
    }
    const playPromise = actionVideo.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {
        errorsEl.textContent = "Action video failed to play.";
        returnToIdle();
      });
    }
  });

  actionVideo.addEventListener("ended", () => {
    returnToIdle();
  });

  actionVideo.addEventListener("error", () => {
    if (hasAction) {
      errorsEl.textContent = "Action video failed to load (unsupported or missing).";
    }
  });
})();
