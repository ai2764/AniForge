(() => {
  const form = document.getElementById("setup-form");
  const imageInput = document.getElementById("image-input");
  const dropZone = document.getElementById("drop-zone");
  const dropZoneText = document.getElementById("drop-zone-text");
  const dropZonePreview = document.getElementById("drop-zone-preview");
  const statusEl = document.getElementById("status");
  const errorsEl = document.getElementById("errors");

  const btnCreate = document.getElementById("btn-create-session");
  const btnRunAll = document.getElementById("btn-run-all");
  const btnExtract = document.getElementById("btn-extract");
  const btnIdle = document.getElementById("btn-idle");
  const btnAction = document.getElementById("btn-action");
  const secJoint = document.getElementById("sec-joint");
  const badgeJoint = document.getElementById("badge-joint");
  const chkJointCarry = document.getElementById("joint-carry");
  const lblJointCarry = document.getElementById("lbl-joint-carry");
  const jointOmega = document.getElementById("joint_omega");
  const jointZeta = document.getElementById("joint_zeta");
  const jointSoft = document.getElementById("joint_soft");
  const jointOmegaLabel = document.getElementById("joint_omega_label");
  const jointZetaLabel = document.getElementById("joint_zeta_label");
  const jointSoftLabel = document.getElementById("joint_soft_label");
  const timeB = document.getElementById("time_b");
  const timeT = document.getElementById("time_t");
  const timeBLabel = document.getElementById("time_b_label");
  const timeTLabel = document.getElementById("time_t_label");
  const btnJointPreview = document.getElementById("btn-joint-preview");
  const vidJointBefore = document.getElementById("vid-joint-before");
  const vidJointAfter = document.getElementById("vid-joint-after");
  const btnScailIdle = document.getElementById("btn-scail-idle");
  const btnScailAction = document.getElementById("btn-scail-action");
  const btnScailDefaults = document.getElementById("btn-scail-defaults");
  const scailIdlePositiveEl = document.getElementById("scail_idle_positive");
  const scailActionPositiveEl = document.getElementById("scail_action_positive");
  const scailNegativeEl = document.getElementById("scail_negative");
  const btnTime = document.getElementById("btn-time");
  const btnBgremove = document.getElementById("btn-bgremove");
  const chkJoint = document.getElementById("overshoot-joint");
  const chkTime = document.getElementById("overshoot-time");
  const chkBgIdle = document.getElementById("bgremove_idle");
  const chkBgAction = document.getElementById("bgremove_action");
  const bgModelSel = document.getElementById("bgremove_model");

  const secExtract = document.getElementById("sec-extract");
  const secIdle = document.getElementById("sec-idle");
  const secAction = document.getElementById("sec-action");
  const secScail = document.getElementById("sec-scail");
  const secTime = document.getElementById("sec-time");
  const secPlay = document.getElementById("sec-play");
  const secBgremove = document.getElementById("sec-bgremove");

  const badgeExtract = document.getElementById("badge-extract");
  const badgeIdle = document.getElementById("badge-idle");
  const badgeAction = document.getElementById("badge-action");
  const badgeScail = document.getElementById("badge-scail");
  const badgeScailIdle = document.getElementById("badge-scail-idle");
  const badgeScailAction = document.getElementById("badge-scail-action");
  const badgeTime = document.getElementById("badge-time");
  const badgePlay = document.getElementById("badge-play");
  const badgeBgremove = document.getElementById("badge-bgremove");

  const imgExtract = document.getElementById("img-extract");
  const boxExtract = document.getElementById("box-extract");
  const vidIdleSkel = document.getElementById("vid-idle-skel");
  const imgIdleSkel = document.getElementById("img-idle-skel");
  const boxIdleSkel = document.getElementById("box-idle-skel");
  const vidIdleVideo = document.getElementById("vid-idle-video");
  const boxIdleVideo = document.getElementById("box-idle-video");
  const vidActionSkel = document.getElementById("vid-action-skel");
  const imgActionSkel = document.getElementById("img-action-skel");
  const boxActionSkel = document.getElementById("box-action-skel");
  const vidActionVideo = document.getElementById("vid-action-video");
  const boxActionVideo = document.getElementById("box-action-video");
  const vidTimeVideo = document.getElementById("vid-time-video");
  const boxTimeVideo = document.getElementById("box-time-video");
  const vidIdleNobg = document.getElementById("vid-idle-nobg");
  const boxIdleNobg = document.getElementById("box-idle-nobg");
  const vidActionNobg = document.getElementById("vid-action-nobg");
  const boxActionNobg = document.getElementById("box-action-nobg");
  const vidUploadNobg = document.getElementById("vid-upload-nobg");
  const boxUploadNobg = document.getElementById("box-upload-nobg");
  const bgremoveLinks = document.getElementById("bgremove-links");
  const bgVideoInput = document.getElementById("bgremove-video");
  const bgVideoName = document.getElementById("bgremove-video-name");
  const timeVideoInput = document.getElementById("time-video");
  const timeVideoName = document.getElementById("time-video-name");

  const preview = document.getElementById("preview");
  const idleVideo = document.getElementById("idleVideo");
  const actionVideo = document.getElementById("actionVideo");

  let runId = null;
  let hasAction = false;
  let idleSkelReady = false;
  let actionSkelReady = false;
  let jointPreviewed = false; // true once a joint-overshoot preview exists this action
  let idleScailReady = false;
  let actionScailReady = false;
  let busy = false;
  let imageNatural = null; // { w, h } from selected file
  let currentRunPose = null;

  const POSE_HINTS = {
    standing:
      "Standing: extract still from image; Kimodo idle/action is free (no pin) so prompts produce different motion.",
    sitting:
      "Sitting: extract pose from image, pin hips for idle/action (keeps seated root; limbs free).",
    lying:
      "Lying: extract pose from image, pin hips for idle/action (keeps lying root; limbs free).",
  };
  const poseHint = document.getElementById("pose-hint");

  function selectedPoseMode() {
    const el = form.querySelector('input[name="pose_mode"]:checked');
    return el ? el.value : "standing";
  }

  function selectedActionPoseMode() {
    const el = document.querySelector('input[name="action_pose_mode"]:checked');
    return el ? el.value : currentRunPose || selectedPoseMode();
  }

  function syncActionPoseMode(mode) {
    if (!mode) return;
    const el = document.querySelector('input[name="action_pose_mode"][value="' + mode + '"]');
    if (el) el.checked = true;
  }

  function syncPoseFromServer(mode) {
    if (!mode) return;
    currentRunPose = mode;
    const el = form.querySelector('input[name="pose_mode"][value="' + mode + '"]');
    if (el) el.checked = true;
    syncActionPoseMode(mode);
    syncPoseUi();
  }

  function jointChecked() {
    return !!(chkJoint && chkJoint.checked);
  }
  // Enable/disable the step-by-step joint-overshoot checkbox (with label dimming).
  function setJointCarryEnabled(on) {
    if (!chkJointCarry) return;
    chkJointCarry.disabled = !on;
    if (lblJointCarry) lblJointCarry.style.opacity = on ? "1" : "0.5";
  }
  function timeChecked() {
    return !!(chkTime && chkTime.checked);
  }

  function showActionSkel(data) {
    const actionSkelUrl =
      data.skeleton ||
      (data.skeleton_png ? String(data.skeleton_png).replace(/\.png(\?.*)?$/i, ".mp4") : null);
    if (actionSkelUrl || data.skeleton_png) {
      showVideo(
        boxActionSkel,
        vidActionSkel,
        actionSkelUrl,
        data.skeleton_png,
        imgActionSkel,
        true
      );
    }
  }

  function syncPoseUi() {
    const mode = selectedPoseMode();
    if (poseHint) poseHint.textContent = POSE_HINTS[mode] || POSE_HINTS.standing;
    const extractHint = document.getElementById("extract-hint");
    if (extractHint) {
      const mode = selectedPoseMode();
      extractHint.textContent =
        mode === "standing"
          ? "HMR still from the image (preview only). Standing idle/action use free Kimodo — no pose pin."
          : "HMR pose from the image, pin hips for Kimodo (limbs free). Review still before Idle.";
    }
  }

  form.querySelectorAll('input[name="pose_mode"]').forEach((el) => {
    el.addEventListener("change", () => {
      syncPoseUi();
      if (!runId) syncActionPoseMode(selectedPoseMode());
    });
  });
  syncPoseUi();

  const scaleInput = document.getElementById("output_scale");
  const scaleLabel = document.getElementById("output_scale_label");
  const sizeReadout = document.getElementById("size-readout");
  const scailOutputScaleInput = document.getElementById("scail_output_scale");
  const scailOutputScaleLabel = document.getElementById("scail_output_scale_label");
  const scailSizeReadout = document.getElementById("scail-size-readout");
  const scailPoseStrengthInput = document.getElementById("scail_pose_strength");
  const scailPoseStrengthLabel = document.getElementById("scail_pose_strength_label");
  const scailCfgInput = document.getElementById("scail_cfg");
  const scailCfgLabel = document.getElementById("scail_cfg_label");
  const idleKeepInput = document.getElementById("idle_motion_keep");
  const idleKeepLabel = document.getElementById("idle_motion_keep_label");
  const actionKeepInput = document.getElementById("action_motion_keep");
  const actionKeepLabel = document.getElementById("action_motion_keep_label");
  const actionDurInput = document.getElementById("action_duration");
  const actionDurLabel = document.getElementById("action_duration_label");

  function idleMotionKeepValue() {
    if (!idleKeepInput) return 0.08;
    let k = parseFloat(idleKeepInput.value);
    if (!(k >= 0)) k = 0.08;
    return Math.max(0, Math.min(1, k));
  }

  function actionMotionKeepValue() {
    if (!actionKeepInput) return 1;
    let k = parseFloat(actionKeepInput.value);
    if (!(k >= 0)) k = 1;
    return Math.max(0, Math.min(1, k));
  }

  function actionDurationValue() {
    if (!actionDurInput) return 2;
    let d = parseFloat(actionDurInput.value);
    if (!(d > 0)) d = 2;
    return Math.max(1, Math.min(3, d));
  }

  function scailOutputScaleValue() {
    if (!scailOutputScaleInput) return 1;
    let s = parseFloat(scailOutputScaleInput.value);
    if (!(s > 0)) s = 1;
    return Math.max(0.25, Math.min(1, s));
  }
  function scailPoseStrengthValue() {
    if (!scailPoseStrengthInput) return 1;
    let s = parseFloat(scailPoseStrengthInput.value);
    if (!(s >= 0)) s = 1;
    return Math.max(0, Math.min(1, s));
  }
  function scailCfgValue() {
    if (!scailCfgInput) return 3;
    let s = parseFloat(scailCfgInput.value);
    if (!(s >= 1)) s = 3;
    return Math.max(1, Math.min(10, s));
  }

  function syncIdleKeepLabel() {
    if (!idleKeepInput || !idleKeepLabel) return;
    idleKeepLabel.textContent = Math.round(idleMotionKeepValue() * 100) + "%";
  }
  function syncActionKeepLabel() {
    if (!actionKeepInput || !actionKeepLabel) return;
    actionKeepLabel.textContent = Math.round(actionMotionKeepValue() * 100) + "%";
  }
  function syncActionDurLabel() {
    if (!actionDurInput || !actionDurLabel) return;
    actionDurLabel.textContent = actionDurationValue().toFixed(1) + "s";
  }
  if (idleKeepInput) {
    idleKeepInput.addEventListener("input", syncIdleKeepLabel);
    syncIdleKeepLabel();
  }
  if (actionKeepInput) {
    actionKeepInput.addEventListener("input", syncActionKeepLabel);
    syncActionKeepLabel();
  }
  if (actionDurInput) {
    actionDurInput.addEventListener("input", syncActionDurLabel);
    syncActionDurLabel();
  }

  // Mirror pipeline.generate._output_size (long_cap=1280, short_cap=720, mult=16)
  function computeOutputSize(w, h, scale) {
    const longCap = 1280;
    const shortCap = 720;
    const mult = 16;
    let s = parseFloat(scale);
    if (!(s > 0)) s = 1;
    s = Math.max(0.25, Math.min(1, s));
    const fit = Math.min(longCap / Math.max(w, h), shortCap / Math.min(w, h)) * s;
    const r = (v) => Math.max(mult, Math.round((v * fit) / mult) * mult);
    return { w: r(w), h: r(h) };
  }

  function updateSizeReadout(serverSize) {
    if (!sizeReadout) return;
    if (!imageNatural) {
      sizeReadout.textContent = "Image: —  →  Video: —";
      return;
    }
    const iw = imageNatural.w;
    const ih = imageNatural.h;
    const scale = scaleInput ? scaleInput.value : 1;
    const out = serverSize
      ? { w: serverSize[0], h: serverSize[1] }
      : computeOutputSize(iw, ih, scale);
    const arImg = (iw / ih).toFixed(3);
    const arOut = (out.w / out.h).toFixed(3);
    sizeReadout.textContent =
      "Image: " +
      iw +
      "×" +
      ih +
      " (AR " +
      arImg +
      ")  →  Video: " +
      out.w +
      "×" +
      out.h +
      " (AR " +
      arOut +
      ", scale " +
      Math.round(parseFloat(scale) * 100) +
      "%)";
  }

  function syncScaleLabel() {
    if (!scaleInput || !scaleLabel) return;
    scaleLabel.textContent = Math.round(parseFloat(scaleInput.value) * 100) + "%";
    updateSizeReadout();
  }
  if (scaleInput) {
    scaleInput.addEventListener("input", syncScaleLabel);
    syncScaleLabel();
  }

  function updateScailSizeReadout() {
    if (!scailSizeReadout) return;
    if (!imageNatural) {
      scailSizeReadout.textContent = "Image: —  →  Video: —";
      return;
    }
    const out = computeOutputSize(imageNatural.w, imageNatural.h, scailOutputScaleValue());
    scailSizeReadout.textContent =
      "Image: " + imageNatural.w + "×" + imageNatural.h +
      "  →  Video: " + out.w + "×" + out.h +
      " (scale " + Math.round(scailOutputScaleValue() * 100) + "%)";
  }
  function syncScailScaleLabel() {
    if (scailOutputScaleLabel) {
      scailOutputScaleLabel.textContent = Math.round(scailOutputScaleValue() * 100) + "%";
    }
    updateScailSizeReadout();
  }
  if (scailOutputScaleInput) {
    scailOutputScaleInput.addEventListener("input", syncScailScaleLabel);
    syncScailScaleLabel();
  }
  function syncScailParamLabels() {
    if (scailPoseStrengthLabel) scailPoseStrengthLabel.textContent = scailPoseStrengthValue().toFixed(2);
    if (scailCfgLabel) scailCfgLabel.textContent = scailCfgValue().toFixed(1);
  }
  if (scailPoseStrengthInput) scailPoseStrengthInput.addEventListener("input", syncScailParamLabels);
  if (scailCfgInput) scailCfgInput.addEventListener("input", syncScailParamLabels);
  syncScailParamLabels();

  // -- image picker -------------------------------------------------------
  dropZone.addEventListener("click", () => imageInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
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
    dropZonePreview.onload = () => {
      imageNatural = {
        w: dropZonePreview.naturalWidth,
        h: dropZonePreview.naturalHeight,
      };
      updateSizeReadout();
      updateScailSizeReadout();
    };
    dropZonePreview.src = url;
    dropZonePreview.style.display = "block";
    dropZoneText.textContent = file.name;
    dropZone.classList.add("has-image");
  }

  // -- section helpers ----------------------------------------------------
  function setBadge(el, text, cls) {
    el.textContent = text;
    el.className = "badge" + (cls ? " " + cls : "");
  }

  function unlock(section, badge, label) {
    section.classList.remove("locked");
    setBadge(badge, label || "ready", "");
  }

  function lock(section, badge, label) {
    section.classList.add("locked");
    setBadge(badge, label || "locked", "");
  }

  function bust(url) {
    if (!url) return url;
    return url + (url.includes("?") ? "&" : "?") + "t=" + Date.now();
  }

  /** Static still only (extract pose). */
  function showStill(box, imgEl, pngUrl) {
    if (!box || !imgEl || !pngUrl) return;
    box.classList.add("show");
    imgEl.src = bust(pngUrl);
    imgEl.style.display = "block";
  }

  /**
   * Motion review: prefer looping video; PNG as fallback if video fails.
   * @param {boolean} motion - if true, show playable video for skeleton animation
   */
  function showVideo(box, video, url, pngUrl, imgEl, motion) {
    if (!url && !pngUrl) return;
    box.classList.add("show");
    const wantMotion = motion !== false;

    if (imgEl) {
      if (pngUrl && (!wantMotion || !url)) {
        imgEl.src = bust(pngUrl);
        imgEl.style.display = "block";
      } else if (pngUrl && wantMotion) {
        // keep still as poster-like fallback under video
        imgEl.src = bust(pngUrl);
        imgEl.style.display = "none";
      } else {
        imgEl.style.display = "none";
      }
    }

    if (video && url && wantMotion) {
      video.style.display = "block";
      video.muted = true;
      video.loop = true;
      video.playsInline = true;
      video.controls = true;
      video.src = bust(url);
      video.load();
      // Prefer video always; only fall back to PNG if the file/codec fails.
      // Do not hide video on autoplay policy rejection — controls remain usable.
      video.play().catch(() => {});
      video.onerror = () => {
        if (imgEl && pngUrl) {
          video.style.display = "none";
          imgEl.style.display = "block";
        }
      };
    } else if (video) {
      video.removeAttribute("src");
      video.style.display = "none";
    }
  }

  function setBusy(on, msg) {
    busy = on;
    btnCreate.disabled = on;
    if (btnRunAll) btnRunAll.disabled = on;
    btnExtract.disabled = on || !runId;
    btnIdle.disabled = on || !runId || secIdle.classList.contains("locked");
    btnAction.disabled = on || !runId || secAction.classList.contains("locked");
    if (btnJointPreview) btnJointPreview.disabled = on || !runId || !actionSkelReady;
    // Carry is usable only after a preview exists; derive it (also dims the label).
    setJointCarryEnabled(!on && jointPreviewed);
    if (btnScailIdle) {
      btnScailIdle.disabled = on || !runId || !idleSkelReady;
    }
    if (btnScailAction) {
      btnScailAction.disabled = on || !runId || !actionSkelReady;
    }
    if (btnTime) {
      // Always unlocked: session run_id and/or uploaded video.
      const hasTimeFile =
        timeVideoInput && timeVideoInput.files && timeVideoInput.files[0];
      btnTime.disabled = on || (!runId && !hasTimeFile);
    }
    if (btnBgremove) {
      // Always available: upload and/or session videos (no pipeline lock).
      btnBgremove.disabled = on;
    }
    if (msg !== undefined) statusEl.textContent = msg;
  }

  function fail(err) {
    errorsEl.textContent = typeof err === "string" ? err : JSON.stringify(err, null, 2);
  }

  function clearErrors() {
    errorsEl.textContent = "";
  }

  /**
   * POST multipart form. Throws only on hard failure.
   * Soft warnings (data.warnings) never throw; non-empty data.errors throws
   * only when HTTP is not ok OR there is no usable success payload.
   */
  async function postForm(url, formData) {
    const res = await fetch(url, { method: "POST", body: formData });
    const text = await res.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch (_) {
      data = { error: text || res.statusText, raw: true };
    }
    const hasErrors = data.errors && Object.keys(data.errors).length > 0;
    // Success signals used by staged endpoints (any one is enough).
    const hasSuccess =
      data.action ||
      data.action_timed ||
      data.action_nobg ||
      data.idle ||
      data.idle_nobg ||
      data.upload_nobg ||
      data.idle_nobg_alpha ||
      data.action_nobg_alpha ||
      data.upload_nobg_alpha ||
      data.action_nobg_webm ||
      data.idle_skel ||
      data.action_skel ||
      data.extract_png ||
      data.extract_skel ||
      data.run_id && (data.time_source || data.step);
    if (!res.ok && !hasSuccess) {
      const msg =
        data.error ||
        data.errors ||
        (Object.keys(data).length ? data : null) ||
        (res.status +
          " " +
          (res.statusText || "error") +
          (text ? ": " + text.slice(0, 200) : ""));
      throw msg;
    }
    // Hard errors only when nothing useful was produced.
    if (hasErrors && !hasSuccess) {
      throw data.errors;
    }
    // Surface soft errors/warnings without failing the step.
    if (hasErrors && hasSuccess) {
      data.warnings = Object.assign({}, data.warnings || {}, data.errors);
    }
    return data;
  }

  function resetDownstreamUi() {
    hasAction = false;
    idleSkelReady = false;
    actionSkelReady = false;
    idleScailReady = false;
    actionScailReady = false;
    preview.style.display = "none";
    preview.classList.remove("has-action");
    [boxExtract, boxIdleSkel, boxIdleVideo, boxActionSkel, boxActionVideo, boxTimeVideo].forEach(
      (b) => {
        if (b) b.classList.remove("show");
      }
    );
    lock(secIdle, badgeIdle, "locked");
    lock(secAction, badgeAction, "locked");
    if (secScail && badgeScail) lock(secScail, badgeScail, "locked");
    if (badgeScailIdle) setBadge(badgeScailIdle, "locked", "");
    if (badgeScailAction) setBadge(badgeScailAction, "locked", "");
    // Time overshoot always unlocked (like bgremove).
    if (secTime) secTime.classList.remove("locked");
    if (badgeTime) setBadge(badgeTime, "ready", "");
    if (btnTime) {
      const hasTimeFile =
        timeVideoInput && timeVideoInput.files && timeVideoInput.files[0];
      btnTime.disabled = !runId && !hasTimeFile;
    }
    lock(secPlay, badgePlay, "locked");
    if (chkJointCarry) chkJointCarry.checked = false;
    jointPreviewed = false;
    setJointCarryEnabled(false);
    if (secJoint && badgeJoint) lock(secJoint, badgeJoint, "locked");
    if (btnJointPreview) btnJointPreview.disabled = true;
    if (vidJointBefore) vidJointBefore.removeAttribute("src");
    if (vidJointAfter) vidJointAfter.removeAttribute("src");
    if (btnScailIdle) btnScailIdle.disabled = true;
    if (btnScailAction) btnScailAction.disabled = true;
    // BG remove stays available (upload and/or session files)
    if (secBgremove) secBgremove.classList.remove("locked");
    if (badgeBgremove) setBadge(badgeBgremove, "ready", "");
    if (btnBgremove) btnBgremove.disabled = false;
    [boxIdleNobg, boxActionNobg, boxUploadNobg].forEach((b) => {
      if (b) b.classList.remove("show");
    });
    if (bgremoveLinks) bgremoveLinks.innerHTML = "";
  }

  function refreshPlayerUnlock() {
    if (idleScailReady || actionScailReady) {
      if (secPlay && badgePlay) unlock(secPlay, badgePlay, "ready");
    }
    // Time overshoot stays unlocked regardless of SCAIL state.
    if (secTime) secTime.classList.remove("locked");
    if (badgeTime && badgeTime.textContent === "locked") {
      setBadge(badgeTime, "ready", "");
    }
    if (btnTime) {
      const hasTimeFile =
        timeVideoInput && timeVideoInput.files && timeVideoInput.files[0];
      btnTime.disabled = !runId && !hasTimeFile;
    }
    if (idleScailReady && actionScailReady && badgeScail) {
      setBadge(badgeScail, "done", "done");
    } else if (idleScailReady || actionScailReady) {
      if (badgeScail) setBadge(badgeScail, "partial", "");
    }
  }

  // BG remove always ready
  if (secBgremove) secBgremove.classList.remove("locked");
  if (badgeBgremove) setBadge(badgeBgremove, "ready", "");
  if (btnBgremove) btnBgremove.disabled = false;
  if (timeVideoInput) {
    timeVideoInput.addEventListener("change", () => {
      const f = timeVideoInput.files && timeVideoInput.files[0];
      if (timeVideoName) {
        timeVideoName.textContent = f
          ? f.name + " (" + Math.round(f.size / 1024) + " KB)"
          : "No file chosen — will use session action_nobg / action.mp4 if present.";
      }
      if (btnTime && !busy) {
        btnTime.disabled = !runId && !f;
      }
    });
  }
  if (bgVideoInput) {
    bgVideoInput.addEventListener("change", () => {
      const f = bgVideoInput.files && bgVideoInput.files[0];
      if (bgVideoName) {
        bgVideoName.textContent = f
          ? f.name + " (" + Math.round(f.size / 1024) + " KB)"
          : "No file chosen — will use session idle/action if checked.";
      }
    });
  }

  function applyActionToPlayer(actionUrl, idleUrl) {
    const hint = document.getElementById("preview-click-hint");
    if (idleUrl) {
      showVideo(boxIdleVideo, vidIdleVideo, idleUrl);
      idleVideo.src = bust(idleUrl);
      idleVideo.style.display = "block";
      preview.style.display = "block";
      idleVideo.play().catch(() => {});
      if (secPlay && badgePlay) unlock(secPlay, badgePlay, "ready");
    }
    if (actionUrl) {
      const url = bust(actionUrl);
      showVideo(boxActionVideo, vidActionVideo, actionUrl);
      if (boxTimeVideo && vidTimeVideo) {
        showVideo(boxTimeVideo, vidTimeVideo, actionUrl);
      }
      // Hard reset so a previous MEDIA_ERR does not stick after time-overshoot.
      actionVideo.pause();
      actionVideo.removeAttribute("src");
      actionVideo.load();
      actionVideo.muted = true;
      actionVideo.playsInline = true;
      actionVideo.src = url;
      actionVideo.load();
      actionVideo.onerror = () => {
        const code = actionVideo.error ? actionVideo.error.code : "?";
        errorsEl.textContent =
          "Action video failed to load (media error " +
          code +
          "). URL: " +
          url +
          " — hard-refresh or re-run time overshoot.";
      };
      actionVideo.onloadeddata = () => {
        // clear sticky error message if reload succeeded
        if (errorsEl.textContent && errorsEl.textContent.indexOf("Action video") === 0) {
          errorsEl.textContent = "";
        }
      };
      hasAction = true;
      preview.classList.add("has-action");
      preview.style.display = "block";
      if (secPlay && badgePlay) unlock(secPlay, badgePlay, "ready");
      if (hint) hint.style.display = "block";
    }
  }

  /** @returns {Promise<object>} session data */
  async function doCreateSession() {
    if (!imageInput.files || !imageInput.files[0]) {
      throw "Please choose an image.";
    }
    const fd = new FormData();
    fd.append("image", imageInput.files[0]);
    fd.append("pose_mode", selectedPoseMode());
    if (scaleInput) fd.append("scale", scaleInput.value);
    const seedValue = document.getElementById("seed").value.trim();
    if (seedValue !== "") fd.append("seed", seedValue);

    const data = await postForm("/session", fd);
    runId = data.run_id;
    if (data.pose_mode) syncActionPoseMode(data.pose_mode);
    if (data.seed !== undefined && data.seed !== null) {
      document.getElementById("seed").value = data.seed;
    }
    if (data.size) updateSizeReadout(data.size);
    resetDownstreamUi();
    unlock(secExtract, badgeExtract, "ready");
    btnExtract.disabled = false;
    return data;
  }

  /** @returns {Promise<object>} */
  async function doExtract() {
    if (!runId) throw "No session — create session first.";
    setBadge(badgeExtract, "running", "running");
    const fd = new FormData();
    fd.append("run_id", runId);
    // Send the currently selected pose so re-extract honors sitting/lying
    // instead of falling back to the session's original (often standing) pose.
    fd.append("pose_mode", selectedPoseMode());
    const data = await postForm("/session/extract", fd);
    if (data.pose_mode) syncPoseFromServer(data.pose_mode);
    if (data.pose_changed) resetDownstreamUi();
    if (data.skipped) {
      setBadge(badgeExtract, "skipped", "done");
    } else {
      setBadge(badgeExtract, "done", "done");
      // Extract: still image only (no video player)
      const still =
        data.skeleton_png ||
        (data.skeleton ? String(data.skeleton).replace(/\.mp4(\?.*)?$/i, ".png") : null);
      if (still) showStill(boxExtract, imgExtract, still);
    }
    unlock(secIdle, badgeIdle, "ready");
    btnIdle.disabled = false;
    // Action does not depend on idle — unlock as soon as extract is ready.
    unlock(secAction, badgeAction, "ready");
    btnAction.disabled = false;
    return data;
  }

  /** @returns {Promise<object>} idle skeleton only */
  async function doIdle() {
    if (!runId) throw "No session.";
    setBadge(badgeIdle, "running", "running");
    const fd = new FormData();
    fd.append("run_id", runId);
    fd.append("idle_prompt", document.getElementById("idle_prompt").value);
    fd.append("idle_motion_keep", String(idleMotionKeepValue()));
    const data = await postForm("/session/idle", fd);
    setBadge(badgeIdle, "done", "done");
    idleSkelReady = true;
    idleScailReady = false; // skeleton changed — re-run SCAIL idle
    if (badgeScailIdle) setBadge(badgeScailIdle, "ready", "");
    // Idle motion review: looping skeleton video (not still).
    const idleSkelUrl =
      data.skeleton ||
      (data.skeleton_png ? String(data.skeleton_png).replace(/\.png(\?.*)?$/i, ".mp4") : null);
    if (idleSkelUrl || data.skeleton_png) {
      showVideo(boxIdleSkel, vidIdleSkel, idleSkelUrl, data.skeleton_png, imgIdleSkel, true);
    }
    // Action may already be unlocked after extract; keep it ready.
    if (secAction && badgeAction) {
      unlock(secAction, badgeAction, "ready");
      btnAction.disabled = false;
    }
    unlockScailSection();
    if (btnScailIdle) {
      btnScailIdle.disabled = false;
      if (badgeScailIdle) setBadge(badgeScailIdle, "ready", "");
    }
    return data;
  }

  /** @returns {Promise<object>} action skeleton only */
  async function doAction() {
    if (!runId) throw "No session.";
    const actionPrompt = document.getElementById("action_prompt").value.trim();
    if (!actionPrompt) throw "Action prompt is required.";
    setBadge(badgeAction, "running", "running");
    const fd = new FormData();
    fd.append("run_id", runId);
    fd.append("action_prompt", actionPrompt);
    fd.append("action_pose_mode", selectedActionPoseMode());
    fd.append("action_motion_keep", String(actionMotionKeepValue()));
    fd.append("action_duration", String(actionDurationValue()));
    const data = await postForm("/session/action", fd);
    if (data.pose_mode) syncPoseFromServer(data.pose_mode);
    setBadge(badgeAction, "done", "done");
    showActionSkel(data);
    actionSkelReady = true;
    actionScailReady = false; // skeleton changed — need SCAIL action again
    // SCAIL action positive inherits the Kimodo action prompt verbatim on each run.
    if (scailActionPositiveEl) scailActionPositiveEl.value = actionPrompt;
    // Fresh action skeleton is plain (no overshoot): reset + enable the toggle.
    if (chkJointCarry) chkJointCarry.checked = false;
    jointPreviewed = false; // new plain skeleton — require a fresh preview before carry
    if (secJoint && badgeJoint) unlock(secJoint, badgeJoint, "ready");
    if (btnJointPreview) btnJointPreview.disabled = false;
    setJointCarryEnabled(false); // carry stays disabled until first preview
    if (vidJointBefore) vidJointBefore.src = bust("/runs/" + runId + "/action_skel.mp4");
    if (vidJointAfter) vidJointAfter.removeAttribute("src");
    unlockScailSection();
    if (btnScailAction) {
      btnScailAction.disabled = false;
      if (badgeScailAction) setBadge(badgeScailAction, "ready", "");
    }
    return data;
  }

  /** Unlock SCAIL section when either skeleton is ready. */
  function unlockScailSection() {
    if ((idleSkelReady || actionSkelReady) && secScail && badgeScail) {
      unlock(secScail, badgeScail, "ready");
    }
  }

  function bust(url) {
    return url + (url.indexOf("?") >= 0 ? "&" : "?") + "t=" + Date.now();
  }

  /** Render the overshot preview (non-destructive) into the right window. */
  async function doJointPreview() {
    if (!runId) throw "No session.";
    if (!actionSkelReady) throw "Run action skeleton first.";
    setBadge(badgeAction, "running", "running");
    const fd = new FormData();
    fd.append("run_id", runId);
    fd.append("mode", "preview");
    if (jointOmega) fd.append("joint_omega", jointOmega.value);
    if (jointZeta) fd.append("joint_zeta", jointZeta.value);
    if (jointSoft) fd.append("joint_soft", jointSoft.value);
    const data = await postForm("/session/joint-overshoot", fd);
    if (vidJointAfter && data.skeleton) vidJointAfter.src = bust(data.skeleton);
    setBadge(badgeAction, "preview", "done");
    return data;
  }

  /** Carry the previewed overshoot into SCAIL's guide, or revert to plain. */
  async function doJointCarry(carry) {
    if (!runId) throw "No session.";
    const fd = new FormData();
    fd.append("run_id", runId);
    fd.append("mode", carry ? "carry" : "uncarry");
    if (carry) {
      if (jointOmega) fd.append("joint_omega", jointOmega.value);
      if (jointZeta) fd.append("joint_zeta", jointZeta.value);
      if (jointSoft) fd.append("joint_soft", jointSoft.value);
    }
    const data = await postForm("/session/joint-overshoot", fd);
    actionScailReady = false;
    unlockScailSection();
    if (btnScailAction) {
      btnScailAction.disabled = false;
      if (badgeScailAction) setBadge(badgeScailAction, "ready", "");
    }
    return data;
  }

  /** Load product SCAIL prompt defaults into the textareas (optional action_prompt embed). */
  async function loadScailDefaults({ force = true } = {}) {
    if (!scailIdlePositiveEl && !scailActionPositiveEl && !scailNegativeEl) return;
    const actionPrompt =
      (document.getElementById("action_prompt") &&
        document.getElementById("action_prompt").value) ||
      "";
    const q = actionPrompt
      ? `?action_prompt=${encodeURIComponent(actionPrompt)}`
      : "";
    const r = await fetch("/api/scail-defaults" + q);
    if (!r.ok) throw `scail defaults HTTP ${r.status}`;
    const d = await r.json();
    if (scailIdlePositiveEl && (force || !scailIdlePositiveEl.value.trim())) {
      scailIdlePositiveEl.value = d.idle_positive || "";
    }
    if (scailActionPositiveEl && (force || !scailActionPositiveEl.value.trim())) {
      scailActionPositiveEl.value = d.action_positive || d.action_positive_base || "";
    }
    if (scailNegativeEl && (force || !scailNegativeEl.value.trim())) {
      scailNegativeEl.value = d.negative || "";
    }
    return d;
  }

  function appendScailPrompts(fd) {
    if (scailIdlePositiveEl) {
      fd.append("scail_idle_positive", scailIdlePositiveEl.value || "");
    }
    if (scailActionPositiveEl) {
      fd.append("scail_action_positive", scailActionPositiveEl.value || "");
    }
    if (scailNegativeEl) {
      fd.append("scail_negative", scailNegativeEl.value || "");
    }
  }

  /** @param {"idle"|"action"|"both"} which
   *  @param {{scale?:number, poseStrength?:number, cfg?:number}} [opts]
   *    step-by-step overrides; Run all omits them so the backend uses defaults */
  async function doScail(which, opts = {}) {
    if (!runId) throw "No session.";
    which = which || "both";
    if (which === "idle" && !idleSkelReady) throw "Run idle skeleton first.";
    if (which === "action" && !actionSkelReady) throw "Run action skeleton first.";
    if (which === "both" && (!idleSkelReady || !actionSkelReady)) {
      throw "Need idle and action skeleton before SCAIL both.";
    }

    const badge =
      which === "idle" ? badgeScailIdle : which === "action" ? badgeScailAction : badgeScail;
    if (badge) setBadge(badge, "running", "running");
    if (badgeScail) setBadge(badgeScail, "running", "running");

    const fd = new FormData();
    fd.append("run_id", runId);
    fd.append("which", which);
    appendScailPrompts(fd);
    // Step-by-step passes explicit overrides; Run all omits them so the backend
    // uses its defaults (Run all resolution / fidelity stay unchanged).
    if (opts.scale != null) fd.append("scale", String(opts.scale));
    if (opts.poseStrength != null) fd.append("pose_strength", String(opts.poseStrength));
    if (opts.cfg != null) fd.append("cfg", String(opts.cfg));
    const data = await postForm("/session/scail", fd);

    if (which === "idle" || which === "both") {
      if (data.idle) {
        idleScailReady = true;
        if (badgeScailIdle) setBadge(badgeScailIdle, "done", "done");
      }
    }
    if (which === "action" || which === "both") {
      if (data.action) {
        actionScailReady = true;
        if (badgeScailAction) setBadge(badgeScailAction, "done", "done");
      }
    }
    if (data.idle_scail_done) idleScailReady = true;
    if (data.action_scail_done) actionScailReady = true;

    applyActionToPlayer(data.action, data.idle);
    refreshPlayerUnlock();
    return data;
  }

  /** Time spring remap: optional upload, else action_nobg, else action.mp4. */
  async function doTimeOvershoot() {
    const hasFile =
      timeVideoInput && timeVideoInput.files && timeVideoInput.files[0];
    if (!runId && !hasFile) {
      throw "Create a session or import a video for time overshoot.";
    }
    if (badgeTime) setBadge(badgeTime, "running", "running");
    statusEl.textContent =
      "Time overshoot running (alpha remaps can take 1–3 min; wait for done)…";
    const fd = new FormData();
    // Only send run_id when it looks like a real session id (avoid "null"/junk).
    if (runId && typeof runId === "string" && runId.length >= 8) {
      fd.append("run_id", runId);
    }
    if (hasFile) fd.append("video", timeVideoInput.files[0]);
    if (timeB) fd.append("time_b", timeB.value);
    if (timeT) fd.append("time_t", timeT.value);
    let data;
    try {
      data = await postForm("/session/time-overshoot", fd);
    } catch (e) {
      // Stale session after server restart — retry once with upload only.
      const msg = typeof e === "string" ? e : JSON.stringify(e);
      if (hasFile && /invalid run_id/i.test(msg)) {
        const fd2 = new FormData();
        fd2.append("video", timeVideoInput.files[0]);
        if (timeB) fd2.append("time_b", timeB.value);
        if (timeT) fd2.append("time_t", timeT.value);
        data = await postForm("/session/time-overshoot", fd2);
        runId = data.run_id || null;
      } else {
        if (badgeTime) setBadge(badgeTime, "error", "");
        throw e;
      }
    }
    if (data.run_id) {
      runId = data.run_id;
    }
    // Mark complete as soon as we have any timed/action output.
    const ok =
      data.action ||
      data.action_timed ||
      data.action_nobg ||
      data.action_timed_webm ||
      data.action_nobg_alpha;
    if (!ok) {
      if (badgeTime) setBadge(badgeTime, "error", "");
      throw data.errors || data.warnings || "Time overshoot produced no output.";
    }
    if (badgeTime) setBadge(badgeTime, "done", "done");
    if (data.warnings && Object.keys(data.warnings).length) {
      errorsEl.textContent =
        "Completed with warnings:\n" + JSON.stringify(data.warnings, null, 2);
    }
    applyActionToPlayer(data.action || data.action_timed, data.idle);
    const previewUrl = data.action_timed || data.action || data.action_nobg;
    if (previewUrl && boxTimeVideo && vidTimeVideo) {
      showVideo(boxTimeVideo, vidTimeVideo, previewUrl);
    }
    if (data.action_nobg && boxActionNobg && vidActionNobg) {
      showVideo(boxActionNobg, vidActionNobg, data.action_nobg);
    }
    unlock(secPlay, badgePlay, "ready");
    // CapCut links if alpha exports exist
    if (bgremoveLinks) {
      const links = [];
      [
        ["action_nobg_alpha", "action_nobg_alpha.mov"],
        ["action_timed_webm", "action_timed.webm"],
        ["action_nobg_webm", "action_nobg.webm"],
        ["action_timed", "action_timed.mp4"],
      ].forEach(([key, name]) => {
        if (data[key]) {
          links.push('<a href="' + data[key] + '" download>' + name + "</a>");
        }
      });
      if (links.length) {
        bgremoveLinks.innerHTML =
          (bgremoveLinks.innerHTML ? bgremoveLinks.innerHTML + " · " : "") +
          "Timed: " +
          links.join(" · ");
      }
    }
    return data;
  }

  function bgremoveWhich() {
    const idle = chkBgIdle && chkBgIdle.checked;
    const act = chkBgAction && chkBgAction.checked;
    const hasFile = bgVideoInput && bgVideoInput.files && bgVideoInput.files[0];
    if (!idle && !act && !hasFile) {
      throw "Import a video, or check session idle/action.";
    }
    if (idle && act) return "both";
    if (idle) return "idle";
    if (act) return "action";
    // upload only
    return "both"; // session flags none; upload handled separately
  }

  async function doBgremove() {
    const hasFile = bgVideoInput && bgVideoInput.files && bgVideoInput.files[0];
    const idle = chkBgIdle && chkBgIdle.checked;
    const act = chkBgAction && chkBgAction.checked;
    if (!hasFile && !idle && !act) {
      throw "Import a video file, and/or check session idle/action.";
    }
    let which = "both";
    if (idle && act) which = "both";
    else if (idle) which = "idle";
    else if (act) which = "action";
    else which = "upload";

    if (badgeBgremove) setBadge(badgeBgremove, "running", "running");
    const fd = new FormData();
    if (runId) fd.append("run_id", runId);
    fd.append("which", which);
    if (bgModelSel) fd.append("model", bgModelSel.value);
    if (hasFile) fd.append("video", bgVideoInput.files[0]);

    const res = await fetch("/session/bgremove", { method: "POST", body: fd });
    const text = await res.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch (_) {
      data = { error: text || res.statusText };
    }
    if (data.run_id && !runId) {
      runId = data.run_id;
    }
    const hasOut =
      data.idle_nobg ||
      data.action_nobg ||
      data.upload_nobg ||
      data.idle_nobg_webm ||
      data.action_nobg_webm ||
      data.upload_nobg_webm ||
      data.idle_nobg_alpha ||
      data.action_nobg_alpha ||
      data.upload_nobg_alpha;
    if (!res.ok && !hasOut) {
      throw data.error || data.errors || data;
    }
    if (data.errors && Object.keys(data.errors).length && hasOut) {
      errorsEl.textContent = JSON.stringify(data.errors, null, 2);
    } else if (data.errors && Object.keys(data.errors).length && !hasOut) {
      throw data.errors;
    }
    if (badgeBgremove) setBadge(badgeBgremove, "done", "done");
    if (data.upload_nobg) {
      showVideo(boxUploadNobg, vidUploadNobg, data.upload_nobg);
    }
    if (data.idle_nobg) {
      showVideo(boxIdleNobg, vidIdleNobg, data.idle_nobg);
    }
    if (data.action_nobg) {
      showVideo(boxActionNobg, vidActionNobg, data.action_nobg);
    }
    applyActionToPlayer(
      data.action_nobg || data.upload_nobg || data.action,
      data.idle_nobg || data.idle
    );
    const links = [];
    [
      // CapCut: ProRes 4444 with real alpha (libvpx-decoded)
      ["upload_nobg_alpha", "upload_nobg_alpha.mov"],
      ["idle_nobg_alpha", "idle_nobg_alpha.mov"],
      ["action_nobg_alpha", "action_nobg_alpha.mov"],
      ["upload_nobg_webm", "upload_nobg.webm"],
      ["idle_nobg_webm", "idle_nobg.webm"],
      ["action_nobg_webm", "action_nobg.webm"],
      ["upload_nobg", "upload_nobg.mp4"],
      ["idle_nobg", "idle_nobg.mp4"],
      ["action_nobg", "action_nobg.mp4"],
    ].forEach(([key, name]) => {
      if (data[key]) {
        links.push('<a href="' + data[key] + '" download>' + name + "</a>");
      }
    });
    if (bgremoveLinks) bgremoveLinks.innerHTML = links.join(" · ");
    unlock(secPlay, badgePlay, "ready");
    return data;
  }

  // -- 1 create session ---------------------------------------------------
  btnCreate.addEventListener("click", async () => {
    if (busy) return;
    clearErrors();
    setBusy(true, "Creating session...");
    try {
      const data = await doCreateSession();
      const sizeTxt = data.size ? data.size[0] + "×" + data.size[1] : "?";
      statusEl.textContent =
        "Session " + runId.slice(0, 8) + "… ready. Output " + sizeTxt + ". Run Extract or Run all.";
    } catch (e) {
      fail(e);
      statusEl.textContent = "Session create failed.";
    } finally {
      setBusy(false);
      btnExtract.disabled = !runId;
    }
  });

  // -- 2 extract ----------------------------------------------------------
  btnExtract.addEventListener("click", async () => {
    if (!runId || busy) return;
    clearErrors();
    setBusy(true, "Extracting skeleton...");
    try {
      const data = await doExtract();
      statusEl.textContent =
        (data.skipped ? "Extract skipped. " : "Extract done. Review skeleton, then Start idle. ") +
        (data.constraint_joints ? "pin=" + JSON.stringify(data.constraint_joints) : "");
    } catch (e) {
      fail(e);
      setBadge(badgeExtract, "error", "");
      statusEl.textContent = "Extract failed.";
    } finally {
      setBusy(false);
    }
  });

  // -- 3 idle skeleton ----------------------------------------------------
  btnIdle.addEventListener("click", async () => {
    if (!runId || busy) return;
    clearErrors();
    setBusy(true, "Running idle skeleton motion (Kimodo)...");
    try {
      const data = await doIdle();
      const stdB = data.motion_std_before != null ? Number(data.motion_std_before).toFixed(4) : "?";
      const stdS = data.motion_std_source != null ? Number(data.motion_std_source).toFixed(4) : "?";
      const stdA = data.motion_std != null ? Number(data.motion_std).toFixed(4) : "?";
      const keepPct =
        data.idle_motion_keep != null
          ? Math.round(Number(data.idle_motion_keep) * 100) + "%"
          : Math.round(idleMotionKeepValue() * 100) + "%";
      const anchor = data.idle_anchored_to_extract ? "extract@100%" : "frame0";
      statusEl.textContent =
        "Idle done (amount=" +
        keepPct +
        ", " +
        anchor +
        ", raw→src→out " +
        stdB +
        "→" +
        stdS +
        "→" +
        stdA +
        "). Review video, then action.";
    } catch (e) {
      fail(e);
      setBadge(badgeIdle, "error", "");
      statusEl.textContent = "Idle skeleton failed.";
    } finally {
      setBusy(false);
    }
  });

  // -- 4 action skeleton --------------------------------------------------
  btnAction.addEventListener("click", async () => {
    if (!runId || busy) return;
    clearErrors();
    setBusy(true, "Running action skeleton motion (Kimodo)...");
    try {
      const act = await doAction();
      const amt =
        act.action_motion_keep != null
          ? Math.round(Number(act.action_motion_keep) * 100) + "%"
          : Math.round(actionMotionKeepValue() * 100) + "%";
      const dur =
        act.action_duration != null
          ? Number(act.action_duration).toFixed(1) + "s"
          : actionDurationValue().toFixed(1) + "s";
      const f0 =
        act.action_f0_err != null ? Number(act.action_f0_err).toFixed(5) : "?";
      const up =
        act.upper_motion_std != null
          ? Number(act.upper_motion_std).toFixed(4)
          : "?";
      const std =
        act.motion_std != null ? Number(act.motion_std).toFixed(4) : "?";
      const lock = act.action_lock_lower ? " legsLocked" : "";
      const msg =
        "Action done (" +
        dur +
        ", amount=" +
        amt +
        ", f0_err=" +
        f0 +
        ", upper_std=" +
        up +
        ", std=" +
        std +
        lock +
        ").";
      if (jointChecked()) {
        statusEl.textContent = msg + " Applying joint overshoot…";
        await doJointCarry(true); // self-springs preview on the server
        statusEl.textContent = "Joint overshoot applied. Review skeleton, then SCAIL2.";
      } else {
        statusEl.textContent = msg + " Optional joint, then SCAIL2.";
      }
    } catch (e) {
      fail(e);
      setBadge(badgeAction, "error", "");
      statusEl.textContent = "Action skeleton failed.";
    } finally {
      setBusy(false);
    }
  });

  // -- 5 Joint overshoot (optional standalone step) -----------------------
  async function runJointPreview() {
    clearErrors();
    setBusy(true, "Rendering overshoot preview…");
    try {
      await doJointPreview();
      jointPreviewed = true;
      setJointCarryEnabled(true); // enable carry once a preview exists
      if (chkJointCarry && chkJointCarry.checked) await doJointCarry(true); // keep guide in sync
      statusEl.textContent = "Overshoot preview ready. Check Carry into SCAIL to use it.";
    } catch (e) {
      fail(e); setBadge(badgeAction, "error", "");
      statusEl.textContent = "Overshoot preview failed.";
    } finally { setBusy(false); }
  }
  if (btnJointPreview) {
    btnJointPreview.addEventListener("click", () => {
      if (!runId || busy) return;
      runJointPreview();
    });
  }
  if (chkJointCarry) {
    chkJointCarry.addEventListener("change", async () => {
      if (!runId || busy) return;
      const carry = chkJointCarry.checked;
      clearErrors();
      setBusy(true, carry ? "Carrying overshoot into SCAIL…" : "Reverting to plain skeleton…");
      try {
        await doJointCarry(carry);
        statusEl.textContent = carry
          ? "Overshoot carried into SCAIL. Re-run SCAIL2 to update character."
          : "Reverted to plain action skeleton. Re-run SCAIL2 to update character.";
      } catch (e) {
        chkJointCarry.checked = !carry;
        fail(e); setBadge(badgeAction, "error", "");
        statusEl.textContent = "Carry toggle failed.";
      } finally { setBusy(false); }
    });
  }
  function wireJointSlider(el, label, fmt) {
    if (!el) return;
    const upd = () => { if (label) label.textContent = fmt(el.value); };
    el.addEventListener("input", upd);
    el.addEventListener("change", () => {
      upd();
      // Re-preview on release once the section is usable; runJointPreview re-carries if checked.
      if (btnJointPreview && !btnJointPreview.disabled && !busy && runId) runJointPreview();
    });
    upd();
  }
  wireJointSlider(jointOmega, jointOmegaLabel, (v) => String(Math.round(Number(v))));
  wireJointSlider(jointZeta, jointZetaLabel, (v) => Number(v).toFixed(2));
  wireJointSlider(jointSoft, jointSoftLabel, (v) => Number(v).toFixed(1));

  function wireValueSlider(el, label, fmt) {
    if (!el) return;
    const update = () => { if (label) label.textContent = fmt(el.value); };
    el.addEventListener("input", update);
    update();
  }
  wireValueSlider(timeB, timeBLabel, (v) => Number(v).toFixed(2));
  wireValueSlider(timeT, timeTLabel, (v) => Number(v).toFixed(2) + "s");

  // -- 5 SCAIL2 character (idle / action separate) ------------------------
  // Fill product defaults once (server is source of truth).
  loadScailDefaults({ force: true }).catch((e) => {
    console.warn("scail defaults", e);
  });
  if (btnScailDefaults) {
    btnScailDefaults.addEventListener("click", async () => {
      if (busy) return;
      try {
        await loadScailDefaults({ force: true });
        statusEl.textContent =
          "SCAIL prompt defaults reloaded (action positive uses current Action prompt).";
      } catch (e) {
        fail(e);
      }
    });
  }
  if (btnScailIdle) {
    btnScailIdle.addEventListener("click", async () => {
      if (!runId || busy) return;
      clearErrors();
      setBusy(true, "SCAIL2 idle: drive character with idle guide…");
      try {
        await doScail("idle", { scale: scailOutputScaleValue(), poseStrength: scailPoseStrengthValue(), cfg: scailCfgValue() });
        statusEl.textContent = "SCAIL idle done. Action SCAIL can run when action skeleton is ready.";
      } catch (e) {
        fail(e);
        if (badgeScailIdle) setBadge(badgeScailIdle, "error", "");
        statusEl.textContent = "SCAIL idle failed.";
      } finally {
        setBusy(false);
      }
    });
  }
  if (btnScailAction) {
    btnScailAction.addEventListener("click", async () => {
      if (!runId || busy) return;
      clearErrors();
      setBusy(true, "SCAIL2 action: drive character with action guide…");
      try {
        await doScail("action", { scale: scailOutputScaleValue(), poseStrength: scailPoseStrengthValue(), cfg: scailCfgValue() });
        statusEl.textContent =
          "SCAIL action done. Next: step 7 bg remove, then optional time overshoot.";
      } catch (e) {
        fail(e);
        if (badgeScailAction) setBadge(badgeScailAction, "error", "");
        statusEl.textContent = "SCAIL action failed.";
      } finally {
        setBusy(false);
      }
    });
  }

  // -- 6 background removal (before time overshoot) -----------------------
  if (btnBgremove) {
    btnBgremove.addEventListener("click", async () => {
      // Allow upload-only without prior session: doBgremove can mint run_id.
      if (busy) return;
      if (!runId && !(bgVideoInput && bgVideoInput.files && bgVideoInput.files[0])) {
        statusEl.textContent = "Create a session or import a video first.";
        return;
      }
      clearErrors();
      setBusy(true, "Background removal (videoBGremoval)… kill Comfy if VRAM is full.");
      try {
        await doBgremove();
        statusEl.textContent =
          "BG removal done. Download *_nobg_alpha.mov for CapCut (real alpha).";
      } catch (e) {
        fail(e);
        if (badgeBgremove) setBadge(badgeBgremove, "error", "");
        statusEl.textContent = "Background removal failed.";
      } finally {
        setBusy(false);
      }
    });
  }

  // -- 7 time overshoot (session video and/or upload) ---------------------
  if (btnTime) {
    btnTime.addEventListener("click", async () => {
      if (busy) return;
      const hasFile =
        timeVideoInput && timeVideoInput.files && timeVideoInput.files[0];
      if (!runId && !hasFile) {
        statusEl.textContent = "Create a session or import a video first.";
        return;
      }
      clearErrors();
      setBusy(
        true,
        hasFile
          ? "Time overshoot on uploaded video…"
          : "Time overshoot (prefers action_nobg, else action.mp4)…"
      );
      try {
        const data = await doTimeOvershoot();
        const alphaHint = data && data.action_nobg_alpha
          ? " CapCut: action_nobg_alpha.mov"
          : data && data.has_alpha
            ? " (alpha webm ready)"
            : "";
        statusEl.textContent =
          "Time overshoot done." + alphaHint + " Preview uses gray-bg mp4 (matches idle).";
        if (badgeTime) setBadge(badgeTime, "done", "done");
      } catch (e) {
        fail(e);
        if (badgeTime) setBadge(badgeTime, "error", "");
        statusEl.textContent = "Time overshoot failed.";
      } finally {
        setBusy(false);
      }
    });
  }

  // -- Mode tabs (Run all / Step by step) --------------------------------
  const tabBtnRunall = document.getElementById("tab-btn-runall");
  const tabBtnSteps = document.getElementById("tab-btn-steps");
  const tabPanelRunall = document.getElementById("tab-runall");
  const tabPanelSteps = document.getElementById("tab-steps");
  const actionPromptEl = document.getElementById("action_prompt");
  const actionPromptStepsEl = document.getElementById("action_prompt_steps");

  function setModeTab(name) {
    const isRun = name === "runall";
    if (tabBtnRunall) tabBtnRunall.setAttribute("aria-selected", isRun ? "true" : "false");
    if (tabBtnSteps) tabBtnSteps.setAttribute("aria-selected", isRun ? "false" : "true");
    if (tabPanelRunall) tabPanelRunall.classList.toggle("active", isRun);
    if (tabPanelSteps) tabPanelSteps.classList.toggle("active", !isRun);
    // Global pose and output scale are Run-all controls. Step by step keeps
    // action pose local to Step 4 and sets size at the SCAIL step.
    const globalPoseSlot = document.getElementById("global-pose-slot");
    if (globalPoseSlot) globalPoseSlot.style.display = isRun ? "" : "none";
    const globalScaleSlot = document.getElementById("global-scale-slot");
    if (globalScaleSlot) globalScaleSlot.style.display = isRun ? "" : "none";
  }

  function syncActionPrompt(from, to) {
    if (!from || !to) return;
    if (to.value !== from.value) to.value = from.value;
  }

  if (tabBtnRunall) {
    tabBtnRunall.addEventListener("click", () => {
      syncActionPrompt(actionPromptStepsEl, actionPromptEl);
      setModeTab("runall");
    });
  }
  if (tabBtnSteps) {
    tabBtnSteps.addEventListener("click", () => {
      syncActionPrompt(actionPromptEl, actionPromptStepsEl);
      setModeTab("steps");
    });
  }
  if (actionPromptEl && actionPromptStepsEl) {
    actionPromptEl.addEventListener("input", () => syncActionPrompt(actionPromptEl, actionPromptStepsEl));
    actionPromptStepsEl.addEventListener("input", () => syncActionPrompt(actionPromptStepsEl, actionPromptEl));
  }

  // -- Run all ------------------------------------------------------------
  if (btnRunAll) {
    btnRunAll.addEventListener("click", async () => {
      if (busy) return;
      clearErrors();
      if (!imageInput.files || !imageInput.files[0]) {
        statusEl.textContent = "Please choose an image.";
        return;
      }
      syncActionPrompt(actionPromptStepsEl, actionPromptEl);
      if (!actionPromptEl || !actionPromptEl.value.trim()) {
        statusEl.textContent = "Action prompt is required for Run all.";
        if (actionPromptEl) actionPromptEl.focus();
        setModeTab("runall");
        return;
      }
      setBusy(true, "Run all: creating session...");
      try {
        const sess = await doCreateSession();
        const sizeTxt = sess.size ? sess.size[0] + "×" + sess.size[1] : "?";
        statusEl.textContent = "Run all: extract pose… (" + sizeTxt + ")";
        await doExtract();
        statusEl.textContent = "Run all: idle skeleton motion…";
        await doIdle();
        statusEl.textContent = "Run all: action skeleton motion…";
        await doAction();
        if (jointChecked()) {
          statusEl.textContent = "Run all: joint overshoot on skeleton…";
          await doJointCarry(true); // self-springs preview on the server
        }
        statusEl.textContent = "Run all: SCAIL idle…";
        await doScail("idle");
        statusEl.textContent = "Run all: SCAIL action…";
        await doScail("action");
        // Step 6 before time overshoot: session idle + action when present.
        if (chkBgIdle) chkBgIdle.checked = true;
        if (chkBgAction) chkBgAction.checked = true;
        statusEl.textContent = "Run all: background removal…";
        await doBgremove();
        if (timeChecked()) {
          statusEl.textContent = "Run all: time overshoot (prefers nobg)…";
          await doTimeOvershoot();
        }
        // Ensure combined player is ready and scrolled into view on Run all.
        refreshPlayerUnlock();
        if (secPlay && badgePlay) unlock(secPlay, badgePlay, "ready");
        if (preview) {
          preview.style.display = "block";
          try {
            secPlay.scrollIntoView({ behavior: "smooth", block: "nearest" });
          } catch (_) {}
        }
        statusEl.textContent =
          "Run all done. Session " +
          (runId ? runId.slice(0, 8) + "…" : "") +
          " — click Preview below to play action.";
      } catch (e) {
        fail(e);
        statusEl.textContent = "Run all stopped with an error.";
      } finally {
        setBusy(false);
      }
    });
  }

  // -- click player -------------------------------------------------------
  function returnToIdle() {
    actionVideo.pause();
    actionVideo.style.display = "none";
    idleVideo.style.display = "block";
    idleVideo.play().catch(() => {});
  }

  preview.addEventListener("click", () => {
    if (!hasAction) return;
    // If a prior load failed, try one reload before giving up.
    if (actionVideo.error) {
      const src = actionVideo.currentSrc || actionVideo.src;
      if (src) {
        actionVideo.src = bust(src.split("?")[0]);
        actionVideo.load();
      }
    }
    idleVideo.pause();
    idleVideo.style.display = "none";
    actionVideo.style.display = "block";
    const tryPlay = () => {
      try {
        actionVideo.currentTime = 0;
      } catch (_) {}
      const p = actionVideo.play();
      if (p && p.catch) {
        p.catch((err) => {
          errorsEl.textContent =
            "Action video failed to play: " +
            (err && err.message ? err.message : String(err)) +
            (actionVideo.error ? " (media " + actionVideo.error.code + ")" : "");
          returnToIdle();
        });
      }
    };
    if (actionVideo.readyState >= 2) {
      tryPlay();
    } else {
      actionVideo.addEventListener("loadeddata", tryPlay, { once: true });
      actionVideo.load();
    }
  });

  actionVideo.addEventListener("ended", returnToIdle);
})();
